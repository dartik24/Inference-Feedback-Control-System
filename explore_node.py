import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import Twist
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

import math
import time
import numpy as np
from std_msgs.msg import String

from std_srvs.srv import Trigger
from control_msgs.action import FollowJointTrajectory

class ExploreNode(Node):
    def __init__(self):
        super().__init__("explore_node")

        self.nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.cmd_vel_pub = self.create_publisher(Twist, "/stretch/cmd_vel", 10)

        self.joint_pose_pub = self.create_publisher(String, "/joint_pose_cmd", 10)

        self.bridge = CvBridge()

        self.model = YOLO("yolov8s.pt")

        self.image_sub = self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.image_callback,
            10
        )

        self.waypoints = [
            (-2.57, -2.76),
            (-0.306, 0.276),
            (-0.459, -3.83),
            (1.22, 2.83),
        ]

        self.position_mode_client = self.create_client(
            Trigger,
            "/switch_to_position_mode"
        )

        self.navigation_mode_client = self.create_client(
            Trigger,
            "/switch_to_navigation_mode"
        )

        self.head_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/stretch_controller/follow_joint_trajectory"
        )

        self.world_state_path = "world_state.txt"

        self.object_counts = {}

        # self.allowed_objects = {
        #     "person",
        #     "chair",
        #     "couch",
        #     "bed",
        #     "dining table",
        #     "tv",
        #     "laptop",
        #     "keyboard",
        #     "mouse",
        #     "remote",
        #     "cell phone",
        #     "book",
        #     "cup",
        #     "bottle",
        #     "backpack",
        #     "blanket",
        #     "handbag",
        #     "sink",
        #     "refrigerator",
        #     "microwave",
        #     "oven",
        #     "toaster",
        #     "potted plant",
        #     "clock",
        # }

        self.confidence_threshold = 0.5
        self.min_seen_count = 5

    def explore(self):
        self.get_logger().info("Starting exploration")
        open(self.world_state_path, "w").close()

        for x, y in self.waypoints:
            success = self.navigate_to(x, y)

            if success:
                self.scan_360()
                self.save_world_state(x, y)

        self.get_logger().info("Exploration complete")

    def switch_to_position_mode(self):
        if not self.position_mode_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Position mode service not available")
            return False

        req = Trigger.Request()
        future = self.position_mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()
        self.get_logger().info(f"Position mode: {result.message}")
        return result.success


    def switch_to_navigation_mode(self):
        if not self.navigation_mode_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Navigation mode service not available")
            return False

        req = Trigger.Request()
        future = self.navigation_mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()
        self.get_logger().info(f"Navigation mode: {result.message}")
        return result.success

    def navigate_to(self, x, y, theta=0.0):
        self.get_logger().info(f"Navigating to ({x}, {y})")

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y

        goal_msg.pose.pose.orientation.z = np.sin(theta / 2.0)
        goal_msg.pose.pose.orientation.w = np.cos(theta / 2.0)

        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Nav2 action server not available!")
            return False

        future = self.nav_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()

        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error("Goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result()

        if result.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Navigation succeeded")
            return True
        else:
            self.get_logger().error(f"Navigation failed with status {result.status}")
            return False

    def scan_360(self):
        self.object_counts.clear()

        self.switch_to_position_mode()
        self.set_head_tilt(-0.38) 

        self.switch_to_navigation_mode()

        twist = Twist()
        twist.angular.z = 0.3

        duration = (2.0 * math.pi) / abs(twist.angular.z)
        start = time.time()

        while time.time() - start < duration:
            self.cmd_vel_pub.publish(twist)
            rclpy.spin_once(self, timeout_sec=0.1)

        self.stop_robot()

        self.switch_to_position_mode()
        self.set_head_tilt(0.0)
        self.switch_to_navigation_mode()

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        detected_objects = self.detect_objects(frame)

        for obj in detected_objects:
            self.object_counts[obj] = self.object_counts.get(obj, 0) + 1

    def detect_objects(self, frame):
        objects = []

        results = self.model(frame, verbose=False)

        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0])

                if confidence < self.confidence_threshold:
                    continue

                class_id = int(box.cls[0])
                label = result.names[class_id]

                # if label not in self.allowed_objects:
                #     continue

                objects.append(label)

        return objects

    def set_head_tilt(self, angle_rad):
        if not self.head_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Head trajectory action server not available")
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ["joint_head_tilt"]

        point = JointTrajectoryPoint()
        point.positions = [angle_rad]
        point.time_from_start.sec = 2

        goal.trajectory.points.append(point)

        future = self.head_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()

        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error("Head tilt goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        self.get_logger().info(f"Head tilt set to {angle_rad}")
        return True

    def stop_robot(self):
        self.cmd_vel_pub.publish(Twist())

    def save_world_state(self, x, y):
        reliable_objects = {
            obj: count
            for obj, count in self.object_counts.items()
            if count >= self.min_seen_count
        }

        with open(self.world_state_path, "a") as f:
            f.write(f"Location: ({x}, {y})\n")
            f.write("Objects seen:\n")

            if len(reliable_objects) == 0:
                f.write("- None\n")
            else:
                for obj, count in sorted(reliable_objects.items()):
                    f.write(f"- {obj} ({count} detections)\n")

            f.write("\n")


def main(args=None):
    rclpy.init(args=args)

    node = ExploreNode()

    try:
        node.explore()
    except KeyboardInterrupt:
        node.stop_robot()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()