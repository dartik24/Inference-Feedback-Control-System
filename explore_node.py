#!/usr/bin/env python3

import math
import time
from collections import deque

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose

import tf2_ros


class ExploreNode(Node):
    def __init__(self):
        super().__init__("explore_node")

# Modify as needed for your exploration parameters
        self.declare_parameter("exploration_threshold", 0.97)
        self.declare_parameter("frontier_min_size", 20)
        self.declare_parameter("planning_rate", 1.0)
        self.declare_parameter("goal_timeout_sec", 120.0)
        self.declare_parameter("distance_weight", 1.0)
        self.declare_parameter("size_weight", 2.0)
        self.declare_parameter("scan_after_goal", True)
        self.declare_parameter("robot_base_frame", "base_link")
        self.declare_parameter("map_frame", "map")

# Read parameters
        self.exploration_threshold = self.get_parameter("exploration_threshold").value
        self.frontier_min_size = self.get_parameter("frontier_min_size").value
        self.planning_rate = self.get_parameter("planning_rate").value
        self.goal_timeout_sec = self.get_parameter("goal_timeout_sec").value
        self.distance_weight = self.get_parameter("distance_weight").value
        self.size_weight = self.get_parameter("size_weight").value
        self.scan_after_goal = self.get_parameter("scan_after_goal").value
        self.robot_base_frame = self.get_parameter("robot_base_frame").value
        self.map_frame = self.get_parameter("map_frame").value

        self.latest_map = None
        self.map_array = None
        self.active_goal = False
        self.goal_start_time = None
        self.goal_handle = None
        self.blacklisted_goals = []

# ROS 2 subscriptions, publishers, and action clients
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            "/map",
            self.map_callback,
            10
        )

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            "/stretch/cmd_vel",
            10
        )

        self.nav_client = ActionClient(
            self,
            NavigateToPose,
            "navigate_to_pose"
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self
        )

        self.timer = self.create_timer(
            1.0 / self.planning_rate,
            self.exploration_loop
        )

        self.get_logger().info("Explore node started.")

    def map_callback(self, msg):
        self.latest_map = msg

        width = msg.info.width
        height = msg.info.height

        data = np.array(msg.data, dtype=np.int16)
        self.map_array = data.reshape((height, width))

    def exploration_loop(self):
        if self.latest_map is None or self.map_array is None:
            self.get_logger().info("Waiting for /map...")
            return

        exploration = self.compute_exploration_coefficient()

        self.get_logger().info(
            f"Exploration coefficient: {exploration:.3f}"
        )

        if exploration >= self.exploration_threshold:
            self.get_logger().info("Exploration threshold reached.")
            self.cancel_current_goal()
            self.stop_robot()
            return

        if self.active_goal:
            if self.goal_timed_out():
                self.get_logger().warn("Goal timed out. Canceling and blacklisting.")
                self.blacklist_current_goal()
                self.cancel_current_goal()
            return

        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            self.get_logger().warn("Could not get robot pose.")
            return

        frontiers = self.find_frontiers()
        clusters = self.cluster_frontiers(frontiers)

        valid_clusters = [
            c for c in clusters
            if len(c) >= self.frontier_min_size
        ]

        if not valid_clusters:
            self.get_logger().warn("No valid frontiers found. Exploration may be complete or stuck.")
            self.stop_robot()
            return

        best_goal = self.choose_best_frontier(valid_clusters, robot_pose)

        if best_goal is None:
            self.get_logger().warn("No reachable non-blacklisted frontier found.")
            self.stop_robot()
            return

        self.send_nav_goal(best_goal[0], best_goal[1])

    def compute_exploration_coefficient(self):
        unknown = np.count_nonzero(self.map_array == -1)
        known = np.count_nonzero(self.map_array != -1)

        total = known + unknown

        if total == 0:
            return 0.0

        return known / total

    def find_frontiers(self):
        frontiers = []

        height, width = self.map_array.shape

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if self.map_array[y, x] != 0:
                    continue

                neighborhood = self.map_array[y - 1:y + 2, x - 1:x + 2]

                if np.any(neighborhood == -1):
                    frontiers.append((x, y))

        return frontiers

    def cluster_frontiers(self, frontiers):
        frontier_set = set(frontiers)
        visited = set()
        clusters = []

        for cell in frontiers:
            if cell in visited:
                continue

            cluster = []
            queue = deque([cell])
            visited.add(cell)

            while queue:
                current = queue.popleft()
                cluster.append(current)

                cx, cy = current

                neighbors = [
                    (cx + 1, cy),
                    (cx - 1, cy),
                    (cx, cy + 1),
                    (cx, cy - 1),
                    (cx + 1, cy + 1),
                    (cx - 1, cy - 1),
                    (cx + 1, cy - 1),
                    (cx - 1, cy + 1),
                ]

                for n in neighbors:
                    if n in frontier_set and n not in visited:
                        visited.add(n)
                        queue.append(n)

            clusters.append(cluster)

        return clusters

    def choose_best_frontier(self, clusters, robot_pose):
        robot_x, robot_y = robot_pose

        best_score = -float("inf")
        best_goal = None

        for cluster in clusters:
            centroid_cell = self.compute_cluster_centroid(cluster)
            world_point = self.map_to_world(
                centroid_cell[0],
                centroid_cell[1]
            )

            if world_point is None:
                continue

            gx, gy = world_point

            if self.is_blacklisted(gx, gy):
                continue

            distance = math.sqrt(
                (gx - robot_x) ** 2 +
                (gy - robot_y) ** 2
            )

            cluster_size = len(cluster)

            score = (
                self.size_weight * cluster_size
                - self.distance_weight * distance
            )

            if score > best_score:
                best_score = score
                best_goal = (gx, gy)

        return best_goal

    def compute_cluster_centroid(self, cluster):
        xs = [p[0] for p in cluster]
        ys = [p[1] for p in cluster]

        cx = int(sum(xs) / len(xs))
        cy = int(sum(ys) / len(ys))

        return cx, cy

    def map_to_world(self, mx, my):
        if self.latest_map is None:
            return None

        resolution = self.latest_map.info.resolution
        origin = self.latest_map.info.origin.position

        wx = origin.x + (mx + 0.5) * resolution
        wy = origin.y + (my + 0.5) * resolution

        return wx, wy

    def world_to_map(self, wx, wy):
        if self.latest_map is None:
            return None

        resolution = self.latest_map.info.resolution
        origin = self.latest_map.info.origin.position

        mx = int((wx - origin.x) / resolution)
        my = int((wy - origin.y) / resolution)

        height, width = self.map_array.shape

        if mx < 0 or my < 0 or mx >= width or my >= height:
            return None

        return mx, my

    def get_robot_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.robot_base_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.5)
            )

            x = transform.transform.translation.x
            y = transform.transform.translation.y

            return x, y

        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return None

    def send_nav_goal(self, x, y):
        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("Nav2 NavigateToPose action server not available.")
            return

        goal_msg = NavigateToPose.Goal()

        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = self.map_frame
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0

        goal_msg.pose.pose.orientation.w = 1.0

        self.get_logger().info(f"Sending exploration goal: x={x:.2f}, y={y:.2f}")

        self.active_goal = True
        self.goal_start_time = time.time()
        self.current_goal_xy = (x, y)

        send_future = self.nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self.nav_feedback_callback
        )

        send_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().warn("Goal rejected by Nav2.")
            self.active_goal = False
            self.blacklist_current_goal()
            return

        self.get_logger().info("Goal accepted by Nav2.")

        self.goal_handle = goal_handle

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)

    def nav_feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        distance = feedback.distance_remaining

        self.get_logger().debug(
            f"Distance remaining: {distance:.2f} m"
        )

    def goal_result_callback(self, future):
        result = future.result()
        status = result.status

        self.get_logger().info(f"Navigation finished with status: {status}")

        self.active_goal = False
        self.goal_handle = None
        self.goal_start_time = None

        if self.scan_after_goal:
            self.rotate_360()

    def cancel_current_goal(self):
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()

        self.active_goal = False
        self.goal_handle = None
        self.goal_start_time = None

    def goal_timed_out(self):
        if self.goal_start_time is None:
            return False

        elapsed = time.time() - self.goal_start_time

        return elapsed > self.goal_timeout_sec

    def blacklist_current_goal(self):
        if hasattr(self, "current_goal_xy"):
            self.blacklisted_goals.append(self.current_goal_xy)

    def is_blacklisted(self, x, y, radius=0.5):
        for bx, by in self.blacklisted_goals:
            dist = math.sqrt((x - bx) ** 2 + (y - by) ** 2)

            if dist < radius:
                return True

        return False

    def rotate_360(self):
        self.get_logger().info("Performing 360 degree scan.")

        twist = Twist()
        twist.angular.z = 0.5

        duration = 2.0 * math.pi / abs(twist.angular.z)
        start = time.time()

        while rclpy.ok() and time.time() - start < duration:
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.05)

        self.stop_robot()

    def stop_robot(self):
        twist = Twist()
        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)

    node = ExploreNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Explore node interrupted.")
    finally:
        node.cancel_current_goal()
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()