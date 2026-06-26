#!/usr/bin/env python3

import math
import time
from collections import deque

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration

from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
)

from action_msgs.msg import GoalStatus
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose

import tf2_ros


class ExploreNode(Node):
    def __init__(self):
        super().__init__("explore_node")

        # ------------------------------------------------------------
        # Core exploration parameters
        # ------------------------------------------------------------

        self.declare_parameter("exploration_threshold", 0.97)
        self.declare_parameter("frontier_min_size", 20)
        self.declare_parameter("planning_rate", 1.0)
        self.declare_parameter("goal_timeout_sec", 120.0)

        # ------------------------------------------------------------
        # Scoring parameters
        #
        # IMPORTANT:
        # distance_weight now REWARDS farther exploration.
        #
        # score = size_weight * sqrt(cluster_size)
        #       + distance_weight * distance_to_goal
        #
        # This prevents nearby huge frontier clusters from always winning.
        # ------------------------------------------------------------

        self.declare_parameter("size_weight", 1.0)
        self.declare_parameter("distance_weight", 5.0)
        self.declare_parameter("min_goal_distance_m", 0.75)

        # ------------------------------------------------------------
        # Optional scan behavior
        #
        # Default is False because your current issue was:
        # nearby goal succeeds immediately -> robot performs 360 scan forever.
        # ------------------------------------------------------------

        self.declare_parameter("scan_after_goal", False)
        self.declare_parameter("scan_angular_speed", 0.5)

        # ------------------------------------------------------------
        # ROS names
        # ------------------------------------------------------------

        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("nav_action_name", "navigate_to_pose")

        # ------------------------------------------------------------
        # TF frames
        # ------------------------------------------------------------

        self.declare_parameter("robot_base_frame", "base_footprint")
        self.declare_parameter("map_frame", "map")

        # ------------------------------------------------------------
        # Safety / goal generation parameters
        # ------------------------------------------------------------

        self.declare_parameter("blacklist_radius", 0.5)
        self.declare_parameter("frontier_clearance_cells", 2)

        self.declare_parameter("goal_inward_offset_m", 0.20)
        self.declare_parameter("goal_search_radius_m", 0.35)
        self.declare_parameter("goal_clearance_m", 0.15)

        # ------------------------------------------------------------
        # Read parameters
        # ------------------------------------------------------------

        self.exploration_threshold = float(
            self.get_parameter("exploration_threshold").value
        )

        self.frontier_min_size = int(
            self.get_parameter("frontier_min_size").value
        )

        self.planning_rate = float(
            self.get_parameter("planning_rate").value
        )

        self.goal_timeout_sec = float(
            self.get_parameter("goal_timeout_sec").value
        )

        self.size_weight = float(
            self.get_parameter("size_weight").value
        )

        self.distance_weight = float(
            self.get_parameter("distance_weight").value
        )

        self.min_goal_distance_m = float(
            self.get_parameter("min_goal_distance_m").value
        )

        self.scan_after_goal = bool(
            self.get_parameter("scan_after_goal").value
        )

        self.scan_angular_speed = float(
            self.get_parameter("scan_angular_speed").value
        )

        self.map_topic = str(
            self.get_parameter("map_topic").value
        )

        self.cmd_vel_topic = str(
            self.get_parameter("cmd_vel_topic").value
        )

        self.nav_action_name = str(
            self.get_parameter("nav_action_name").value
        )

        self.robot_base_frame = str(
            self.get_parameter("robot_base_frame").value
        )

        self.map_frame = str(
            self.get_parameter("map_frame").value
        )

        self.blacklist_radius = float(
            self.get_parameter("blacklist_radius").value
        )

        self.frontier_clearance_cells = int(
            self.get_parameter("frontier_clearance_cells").value
        )

        self.goal_inward_offset_m = float(
            self.get_parameter("goal_inward_offset_m").value
        )

        self.goal_search_radius_m = float(
            self.get_parameter("goal_search_radius_m").value
        )

        self.goal_clearance_m = float(
            self.get_parameter("goal_clearance_m").value
        )

        # ------------------------------------------------------------
        # Internal state
        # ------------------------------------------------------------

        self.latest_map = None
        self.map_array = None

        self.active_goal = False
        self.goal_handle = None
        self.goal_start_time = None
        self.current_goal_xy = None

        self.blacklisted_goals = []
        self.exploration_finished = False

        # ------------------------------------------------------------
        # QoS for map
        #
        # /map from SLAM/Nav2 often uses transient local durability.
        # This QoS allows the node to receive the latest map immediately.
        # ------------------------------------------------------------

        map_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            map_qos,
        )

        # ------------------------------------------------------------
        # Velocity publisher
        #
        # This is only used by this node for stop commands and optional scans.
        # Nav2 has its own cmd_vel output.
        # ------------------------------------------------------------

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            self.cmd_vel_topic,
            10,
        )

        # ------------------------------------------------------------
        # Nav2 action client
        # ------------------------------------------------------------

        self.nav_client = ActionClient(
            self,
            NavigateToPose,
            self.nav_action_name,
        )

        # ------------------------------------------------------------
        # TF listener
        # ------------------------------------------------------------

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self,
        )

        # ------------------------------------------------------------
        # Main timer
        # ------------------------------------------------------------

        self.timer = self.create_timer(
            1.0 / self.planning_rate,
            self.exploration_loop,
        )

        self.get_logger().info("Explore node started.")
        self.get_logger().info(f"Map topic: {self.map_topic}")
        self.get_logger().info(f"Cmd vel topic: {self.cmd_vel_topic}")
        self.get_logger().info(f"Nav action: {self.nav_action_name}")
        self.get_logger().info(
            f"TF chain expected: {self.map_frame} -> odom -> {self.robot_base_frame}"
        )
        self.get_logger().info(
            f"Scoring: score = {self.size_weight} * sqrt(cluster_size) "
            f"+ {self.distance_weight} * distance"
        )
        self.get_logger().info(
            f"Minimum goal distance: {self.min_goal_distance_m:.2f} m"
        )

    # ------------------------------------------------------------
    # Map callback
    # ------------------------------------------------------------

    def map_callback(self, msg):
        self.latest_map = msg

        width = msg.info.width
        height = msg.info.height

        data = np.array(msg.data, dtype=np.int16)
        self.map_array = data.reshape((height, width))

        unknown = np.count_nonzero(self.map_array == -1)
        free = np.count_nonzero(self.map_array == 0)
        occupied = np.count_nonzero(self.map_array > 0)

        self.get_logger().info(
            f"Received map: {width}x{height}, "
            f"resolution={msg.info.resolution:.4f}, "
            f"unknown={unknown}, free={free}, occupied={occupied}"
        )

    # ------------------------------------------------------------
    # Main exploration loop
    # ------------------------------------------------------------

    def exploration_loop(self):
        if self.exploration_finished:
            return

        if self.latest_map is None or self.map_array is None:
            self.get_logger().info("Waiting for map...")
            return

        exploration = self.compute_exploration_coefficient()
        unknown_count = np.count_nonzero(self.map_array == -1)

        self.get_logger().info(
            f"Exploration coefficient: {exploration:.3f}"
        )

        if unknown_count > 0 and exploration >= self.exploration_threshold:
            self.get_logger().info(
                f"Exploration threshold reached: "
                f"{exploration:.3f} >= {self.exploration_threshold:.3f}"
            )

            self.cancel_current_goal()
            self.stop_robot()
            self.exploration_finished = True
            return

        if unknown_count == 0:
            self.get_logger().warn(
                "Map has zero unknown cells. Frontier exploration may not work."
            )

        if self.active_goal:
            if self.goal_timed_out():
                self.get_logger().warn(
                    "Current goal timed out. Canceling and blacklisting."
                )

                self.blacklist_current_goal()
                self.cancel_current_goal()

            return

        robot_pose = self.get_robot_pose()

        if robot_pose is None:
            self.get_logger().warn("Could not get robot pose.")
            return

        frontiers = self.find_frontiers()

        self.get_logger().info(
            f"Detected {len(frontiers)} raw frontier cells."
        )

        if len(frontiers) == 0:
            self.get_logger().warn(
                "No frontiers found. Exploration may be complete."
            )
            self.stop_robot()
            return

        clusters = self.cluster_frontiers(frontiers)

        valid_clusters = [
            cluster for cluster in clusters
            if len(cluster) >= self.frontier_min_size
        ]

        self.get_logger().info(
            f"Frontier clusters: {len(clusters)}, "
            f"valid clusters: {len(valid_clusters)}"
        )

        if not valid_clusters:
            self.get_logger().warn(
                "No valid frontier clusters found after size filtering."
            )
            self.stop_robot()
            return

        best_goal = self.choose_best_frontier(
            valid_clusters,
            robot_pose,
        )

        if best_goal is None:
            self.get_logger().warn(
                "No safe reachable non-blacklisted frontier goal found."
            )
            self.stop_robot()
            return

        self.send_nav_goal(best_goal[0], best_goal[1])

    # ------------------------------------------------------------
    # Exploration coefficient
    # ------------------------------------------------------------

    def compute_exploration_coefficient(self):
        unknown = np.count_nonzero(self.map_array == -1)
        known = np.count_nonzero(self.map_array != -1)

        total = known + unknown

        if total == 0:
            return 0.0

        return known / total

    # ------------------------------------------------------------
    # Frontier detection
    #
    # Frontier = known free cell adjacent to unknown space.
    # ------------------------------------------------------------

    def find_frontiers(self):
        frontiers = []

        height, width = self.map_array.shape

        for y in range(1, height - 1):
            for x in range(1, width - 1):

                if self.map_array[y, x] != 0:
                    continue

                if not self.has_obstacle_clearance(x, y):
                    continue

                neighborhood = self.map_array[y - 1:y + 2, x - 1:x + 2]

                if np.any(neighborhood == -1):
                    frontiers.append((x, y))

        return frontiers

    # ------------------------------------------------------------
    # Frontier obstacle clearance
    # ------------------------------------------------------------

    def has_obstacle_clearance(self, x, y):
        r = self.frontier_clearance_cells

        height, width = self.map_array.shape

        x_min = max(0, x - r)
        x_max = min(width, x + r + 1)

        y_min = max(0, y - r)
        y_max = min(height, y + r + 1)

        patch = self.map_array[y_min:y_max, x_min:x_max]

        if np.any(patch > 50):
            return False

        return True

    # ------------------------------------------------------------
    # Connected-component clustering
    # ------------------------------------------------------------

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

                for neighbor in neighbors:
                    if neighbor in frontier_set and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            clusters.append(cluster)

        return clusters

    # ------------------------------------------------------------
    # Choose best frontier
    #
    # New behavior:
    # - Skip goals too close to robot.
    # - Reward farther goals.
    # - Still reward larger frontier clusters.
    # ------------------------------------------------------------

    def choose_best_frontier(self, clusters, robot_pose):
        robot_x, robot_y = robot_pose

        robot_cell = self.world_to_map(robot_x, robot_y)

        if robot_cell is None:
            self.get_logger().warn("Robot pose is outside map bounds.")
            return None

        reachable_mask = self.compute_reachable_free_mask(robot_cell)

        best_score = -float("inf")
        best_goal = None

        for cluster in clusters:
            safe_goal_cell = self.find_safe_goal_for_cluster(
                cluster,
                robot_cell,
                reachable_mask,
            )

            if safe_goal_cell is None:
                self.get_logger().warn(
                    f"Skipping frontier cluster of size {len(cluster)} "
                    "because no safe reachable goal was found."
                )
                continue

            world_point = self.map_to_world(
                safe_goal_cell[0],
                safe_goal_cell[1],
            )

            if world_point is None:
                continue

            goal_x, goal_y = world_point

            if self.is_blacklisted(goal_x, goal_y):
                self.get_logger().warn(
                    f"Skipping blacklisted goal: x={goal_x:.2f}, y={goal_y:.2f}"
                )
                continue

            distance = math.sqrt(
                (goal_x - robot_x) ** 2 +
                (goal_y - robot_y) ** 2
            )

            if distance < self.min_goal_distance_m:
                self.get_logger().warn(
                    f"Skipping frontier goal because it is too close: "
                    f"x={goal_x:.2f}, y={goal_y:.2f}, "
                    f"distance={distance:.2f} m"
                )
                continue

            cluster_size = len(cluster)

            size_score = self.size_weight * math.sqrt(cluster_size)
            distance_score = self.distance_weight * distance

            score = size_score + distance_score

            self.get_logger().info(
                f"Frontier safe candidate: "
                f"x={goal_x:.2f}, y={goal_y:.2f}, "
                f"size={cluster_size}, "
                f"distance={distance:.2f}, "
                f"size_score={size_score:.2f}, "
                f"distance_score={distance_score:.2f}, "
                f"score={score:.2f}"
            )

            if score > best_score:
                best_score = score
                best_goal = (goal_x, goal_y)

        if best_goal is not None:
            self.get_logger().info(
                f"Selected FAR frontier goal: "
                f"x={best_goal[0]:.2f}, y={best_goal[1]:.2f}, "
                f"score={best_score:.2f}"
            )

        return best_goal

    # ------------------------------------------------------------
    # Find safe goal for cluster
    # ------------------------------------------------------------

    def find_safe_goal_for_cluster(self, cluster, robot_cell, reachable_mask):
        if self.latest_map is None or self.map_array is None:
            return None

        resolution = self.latest_map.info.resolution

        inward_offset_cells = max(
            1,
            int(self.goal_inward_offset_m / resolution)
        )

        search_radius_cells = max(
            1,
            int(self.goal_search_radius_m / resolution)
        )

        clearance_cells = max(
            1,
            int(self.goal_clearance_m / resolution)
        )

        robot_mx, robot_my = robot_cell

        centroid = self.compute_cluster_centroid(cluster)
        cx, cy = centroid

        sorted_frontier_cells = sorted(
            cluster,
            key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2
        )

        best_cell = None
        best_score = -float("inf")

        for fx, fy in sorted_frontier_cells:

            # Direction from frontier cell toward robot.
            # This moves the goal inward into known free space.
            dx = robot_mx - fx
            dy = robot_my - fy

            norm = math.sqrt(dx * dx + dy * dy)

            if norm < 1e-6:
                continue

            ux = dx / norm
            uy = dy / norm

            base_x = int(round(fx + ux * inward_offset_cells))
            base_y = int(round(fy + uy * inward_offset_cells))

            for sy in range(
                base_y - search_radius_cells,
                base_y + search_radius_cells + 1
            ):
                for sx in range(
                    base_x - search_radius_cells,
                    base_x + search_radius_cells + 1
                ):

                    if not self.is_cell_in_bounds(sx, sy):
                        continue

                    search_dist = math.sqrt(
                        (sx - base_x) ** 2 +
                        (sy - base_y) ** 2
                    )

                    if search_dist > search_radius_cells:
                        continue

                    if not self.is_safe_goal_cell(
                        sx,
                        sy,
                        clearance_cells,
                        reachable_mask,
                    ):
                        continue

                    frontier_dist = math.sqrt(
                        (sx - fx) ** 2 +
                        (sy - fy) ** 2
                    )

                    robot_dist = math.sqrt(
                        (sx - robot_mx) ** 2 +
                        (sy - robot_my) ** 2
                    )

                    # For the safe cell within this cluster:
                    # - prefer close to frontier
                    # - prefer farther from robot slightly
                    local_score = -frontier_dist + 0.05 * robot_dist

                    if local_score > best_score:
                        best_score = local_score
                        best_cell = (sx, sy)

        return best_cell

    # ------------------------------------------------------------
    # Safe goal check
    # ------------------------------------------------------------

    def is_safe_goal_cell(self, mx, my, clearance_cells, reachable_mask):
        if not self.is_cell_in_bounds(mx, my):
            return False

        if self.map_array[my, mx] != 0:
            return False

        if not reachable_mask[my, mx]:
            return False

        height, width = self.map_array.shape

        x_min = max(0, mx - clearance_cells)
        x_max = min(width, mx + clearance_cells + 1)

        y_min = max(0, my - clearance_cells)
        y_max = min(height, my + clearance_cells + 1)

        patch = self.map_array[y_min:y_max, x_min:x_max]

        if np.any(patch > 50):
            return False

        return True

    # ------------------------------------------------------------
    # Reachability flood-fill
    # ------------------------------------------------------------

    def compute_reachable_free_mask(self, robot_cell):
        height, width = self.map_array.shape

        reachable = np.zeros((height, width), dtype=bool)

        start_x, start_y = robot_cell

        if not self.is_cell_in_bounds(start_x, start_y):
            return reachable

        if self.map_array[start_y, start_x] != 0:
            self.get_logger().warn(
                "Robot's current map cell is not marked free. "
                "Trying nearest free cell."
            )

            nearest = self.find_nearest_free_cell(start_x, start_y)

            if nearest is None:
                self.get_logger().warn(
                    "Could not find nearby free cell for reachability flood-fill."
                )
                return reachable

            start_x, start_y = nearest

        queue = deque()
        queue.append((start_x, start_y))
        reachable[start_y, start_x] = True

        while queue:
            x, y = queue.popleft()

            neighbors = [
                (x + 1, y),
                (x - 1, y),
                (x, y + 1),
                (x, y - 1),
            ]

            for nx, ny in neighbors:
                if not self.is_cell_in_bounds(nx, ny):
                    continue

                if reachable[ny, nx]:
                    continue

                if self.map_array[ny, nx] != 0:
                    continue

                reachable[ny, nx] = True
                queue.append((nx, ny))

        return reachable

    # ------------------------------------------------------------
    # Find nearest free cell if robot cell is not exactly free
    # ------------------------------------------------------------

    def find_nearest_free_cell(self, start_x, start_y, max_radius=10):
        for radius in range(1, max_radius + 1):
            for y in range(start_y - radius, start_y + radius + 1):
                for x in range(start_x - radius, start_x + radius + 1):

                    if not self.is_cell_in_bounds(x, y):
                        continue

                    if self.map_array[y, x] == 0:
                        return x, y

        return None

    # ------------------------------------------------------------
    # Cell bounds
    # ------------------------------------------------------------

    def is_cell_in_bounds(self, mx, my):
        height, width = self.map_array.shape

        if mx < 0:
            return False

        if my < 0:
            return False

        if mx >= width:
            return False

        if my >= height:
            return False

        return True

    # ------------------------------------------------------------
    # Cluster centroid
    # ------------------------------------------------------------

    def compute_cluster_centroid(self, cluster):
        xs = [cell[0] for cell in cluster]
        ys = [cell[1] for cell in cluster]

        cx = int(sum(xs) / len(xs))
        cy = int(sum(ys) / len(ys))

        return cx, cy

    # ------------------------------------------------------------
    # Map cell to world
    # ------------------------------------------------------------

    def map_to_world(self, mx, my):
        if self.latest_map is None:
            return None

        resolution = self.latest_map.info.resolution
        origin = self.latest_map.info.origin.position

        wx = origin.x + (mx + 0.5) * resolution
        wy = origin.y + (my + 0.5) * resolution

        return wx, wy

    # ------------------------------------------------------------
    # World to map cell
    # ------------------------------------------------------------

    def world_to_map(self, wx, wy):
        if self.latest_map is None or self.map_array is None:
            return None

        resolution = self.latest_map.info.resolution
        origin = self.latest_map.info.origin.position

        mx = int((wx - origin.x) / resolution)
        my = int((wy - origin.y) / resolution)

        if not self.is_cell_in_bounds(mx, my):
            return None

        return mx, my

    # ------------------------------------------------------------
    # Robot pose from TF
    # ------------------------------------------------------------

    def get_robot_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.robot_base_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.5),
            )

            x = transform.transform.translation.x
            y = transform.transform.translation.y

            return x, y

        except Exception as e:
            self.get_logger().warn(
                f"TF lookup failed from {self.map_frame} "
                f"to {self.robot_base_frame}: {e}"
            )
            return None

    # ------------------------------------------------------------
    # Yaw to quaternion
    # ------------------------------------------------------------

    def yaw_to_quaternion(self, yaw):
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)

        return qz, qw

    # ------------------------------------------------------------
    # Send Nav2 goal
    # ------------------------------------------------------------

    def send_nav_goal(self, x, y):
        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error(
                f"Nav2 NavigateToPose action server not available: "
                f"{self.nav_action_name}"
            )
            return

        goal_msg = NavigateToPose.Goal()

        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = self.map_frame
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.position.z = 0.0

        robot_pose = self.get_robot_pose()

        if robot_pose is not None:
            robot_x, robot_y = robot_pose
            yaw = math.atan2(y - robot_y, x - robot_x)
        else:
            yaw = 0.0

        qz, qw = self.yaw_to_quaternion(yaw)

        goal_msg.pose.pose.orientation.x = 0.0
        goal_msg.pose.pose.orientation.y = 0.0
        goal_msg.pose.pose.orientation.z = qz
        goal_msg.pose.pose.orientation.w = qw

        self.get_logger().info(
            f"Sending Nav2 goal: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}"
        )

        self.active_goal = True
        self.goal_start_time = time.time()
        self.current_goal_xy = (x, y)

        send_future = self.nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self.nav_feedback_callback,
        )

        send_future.add_done_callback(self.goal_response_callback)

    # ------------------------------------------------------------
    # Goal response callback
    # ------------------------------------------------------------

    def goal_response_callback(self, future):
        try:
            goal_handle = future.result()

        except Exception as e:
            self.get_logger().error(f"Goal response failed: {e}")
            self.active_goal = False
            self.goal_handle = None
            self.goal_start_time = None
            return

        if not goal_handle.accepted:
            self.get_logger().warn("Goal rejected by Nav2.")

            self.blacklist_current_goal()

            self.active_goal = False
            self.goal_handle = None
            self.goal_start_time = None
            return

        self.get_logger().info("Goal accepted by Nav2.")

        self.goal_handle = goal_handle

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)

    # ------------------------------------------------------------
    # Nav2 feedback
    # ------------------------------------------------------------

    def nav_feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback

        try:
            distance = feedback.distance_remaining

            self.get_logger().debug(
                f"Distance remaining: {distance:.2f} m"
            )

        except Exception:
            pass

    # ------------------------------------------------------------
    # Nav2 result callback
    # ------------------------------------------------------------

    def goal_result_callback(self, future):
        try:
            result = future.result()

        except Exception as e:
            self.get_logger().error(f"Goal result failed: {e}")
            self.active_goal = False
            self.goal_handle = None
            self.goal_start_time = None
            return

        status = result.status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Navigation goal succeeded.")

            self.active_goal = False
            self.goal_handle = None
            self.goal_start_time = None

            if self.scan_after_goal and not self.exploration_finished:
                self.rotate_360()

        else:
            self.get_logger().warn(
                f"Navigation goal failed or was canceled. Status: {status}"
            )

            self.blacklist_current_goal()

            self.active_goal = False
            self.goal_handle = None
            self.goal_start_time = None

    # ------------------------------------------------------------
    # Cancel current Nav2 goal
    # ------------------------------------------------------------

    def cancel_current_goal(self):
        if self.goal_handle is not None:
            self.get_logger().info("Canceling current Nav2 goal.")
            self.goal_handle.cancel_goal_async()

        self.active_goal = False
        self.goal_handle = None
        self.goal_start_time = None

    # ------------------------------------------------------------
    # Goal timeout check
    # ------------------------------------------------------------

    def goal_timed_out(self):
        if self.goal_start_time is None:
            return False

        elapsed = time.time() - self.goal_start_time

        return elapsed > self.goal_timeout_sec

    # ------------------------------------------------------------
    # Blacklist current goal
    # ------------------------------------------------------------

    def blacklist_current_goal(self):
        if self.current_goal_xy is None:
            return

        self.get_logger().warn(
            f"Blacklisting goal: "
            f"x={self.current_goal_xy[0]:.2f}, "
            f"y={self.current_goal_xy[1]:.2f}"
        )

        self.blacklisted_goals.append(self.current_goal_xy)

    # ------------------------------------------------------------
    # Is goal blacklisted?
    # ------------------------------------------------------------

    def is_blacklisted(self, x, y):
        for bx, by in self.blacklisted_goals:
            dist = math.sqrt(
                (x - bx) ** 2 +
                (y - by) ** 2
            )

            if dist < self.blacklist_radius:
                return True

        return False

    # ------------------------------------------------------------
    # Optional 360-degree scan
    # ------------------------------------------------------------

    def rotate_360(self):
        self.get_logger().info("Performing 360-degree scan.")

        if abs(self.scan_angular_speed) < 1e-6:
            self.get_logger().warn(
                "scan_angular_speed is too small. Skipping scan."
            )
            return

        twist = Twist()
        twist.angular.z = self.scan_angular_speed

        duration = 2.0 * math.pi / abs(self.scan_angular_speed)
        start = time.time()

        while rclpy.ok() and time.time() - start < duration:
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.05)

        self.stop_robot()

        self.get_logger().info("360-degree scan complete.")

    # ------------------------------------------------------------
    # Stop robot
    # ------------------------------------------------------------

    def stop_robot(self):
        twist = Twist()
        self.cmd_vel_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)

    node = ExploreNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info("Explore node interrupted by user.")

    finally:
        node.cancel_current_goal()
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()