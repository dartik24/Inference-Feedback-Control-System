#!/usr/bin/env python3

import math
import os
import time

import numpy as np

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from std_srvs.srv import Trigger

import tf2_ros


class ExploreNode3D(Node):
    """Explore with Stretch funmap while accumulating a global 3D voxel map.

    Goal selection is based only on the accumulated 3D voxel map. Funmap owns
    the Stretch navigation action and the head-scan-based map updates.
    """

    def __init__(self):
        super().__init__("explore_node_3d")

        self.declare_parameters(
            namespace="",
            parameters=[
                ("pointcloud_topic", "/funmap/point_cloud2"),
                ("map_cloud_topic", "/explore_3d/map_cloud"),
                ("cmd_vel_topic", "/cmd_vel"),
                ("scan_cmd_vel_topic", "/stretch/cmd_vel"),
                ("nav_action_name", "/move_base"),
                ("use_funmap", True),
                ("funmap_head_scan_service", "/funmap/trigger_head_scan"),
                ("funmap_service_timeout_sec", 0.2),
                ("head_scan_before_first_goal", True),
                ("map_frame", "map"),
                ("robot_base_frame", "base_link"),
                ("camera_frame_fallback", ""),
                ("planning_rate", 0.5),
                ("cloud_publish_rate", 1.0),
                ("min_goal_distance_m", 0.20),
                ("ideal_goal_distance_m", 0.85),
                ("max_goal_distance_m", 2.50),
                ("goal_timeout_sec", 90.0),
                ("blacklist_radius_m", 0.25),
                ("max_candidates_to_test", 30),
                ("candidate_angle_count", 32),
                ("candidate_distance_step_m", 0.25),
                ("robot_clearance_radius_m", 0.35),
                ("corridor_clearance_radius_m", 0.22),
                ("obstacle_min_z_m", 0.02),
                ("obstacle_max_z_m", 1.20),
                ("frontier_band_width_m", 0.35),
                ("frontier_sparse_radius_m", 0.45),
                ("min_3d_voxels_for_navigation", 80),
                ("voxel_size_m", 0.05),
                ("filter_point_range", False),
                ("min_point_range_m", 0.25),
                ("max_point_range_m", 20.0),
                ("min_z_m", -0.20),
                ("max_z_m", 2.20),
                ("max_points_per_cloud", 12000),
                ("max_voxels", 800000),
                ("save_map", True),
                ("save_period_sec", 30.0),
                ("save_directory", "/tmp/stretch3_explore_3d"),
                ("global_map_filename", "stretch3_global_voxel_map.ply"),
                ("load_existing_map", True),
                ("scan_after_goal", True),
                ("scan_after_failed_goal", True),
                ("scan_angular_speed", 0.35),
                ("scan_publish_rate", 20.0),
                ("scan_timeout_padding_sec", 2.0),
            ],
        )

        self.pointcloud_topic = self.get_string("pointcloud_topic")
        self.map_cloud_topic = self.get_string("map_cloud_topic")
        self.cmd_vel_topic = self.get_string("cmd_vel_topic")
        self.scan_cmd_vel_topic = self.get_string("scan_cmd_vel_topic")
        self.nav_action_name = self.get_string("nav_action_name")
        self.use_funmap = self.get_bool("use_funmap")
        self.funmap_head_scan_service = self.get_string("funmap_head_scan_service")
        self.funmap_service_timeout_sec = self.get_float("funmap_service_timeout_sec")
        self.head_scan_before_first_goal = self.get_bool("head_scan_before_first_goal")
        self.map_frame = self.get_string("map_frame")
        self.robot_base_frame = self.get_string("robot_base_frame")
        self.camera_frame_fallback = self.get_string("camera_frame_fallback")

        self.planning_rate = self.get_float("planning_rate")
        self.cloud_publish_rate = self.get_float("cloud_publish_rate")
        self.min_goal_distance_m = self.get_float("min_goal_distance_m")
        self.ideal_goal_distance_m = self.get_float("ideal_goal_distance_m")
        self.max_goal_distance_m = self.get_float("max_goal_distance_m")
        self.goal_timeout_sec = self.get_float("goal_timeout_sec")
        self.blacklist_radius_m = self.get_float("blacklist_radius_m")
        self.max_candidates_to_test = self.get_int("max_candidates_to_test")
        self.candidate_angle_count = self.get_int("candidate_angle_count")
        self.candidate_distance_step_m = self.get_float("candidate_distance_step_m")
        self.robot_clearance_radius_m = self.get_float("robot_clearance_radius_m")
        self.corridor_clearance_radius_m = self.get_float("corridor_clearance_radius_m")
        self.obstacle_min_z_m = self.get_float("obstacle_min_z_m")
        self.obstacle_max_z_m = self.get_float("obstacle_max_z_m")
        self.frontier_band_width_m = self.get_float("frontier_band_width_m")
        self.frontier_sparse_radius_m = self.get_float("frontier_sparse_radius_m")
        self.min_3d_voxels_for_navigation = self.get_int("min_3d_voxels_for_navigation")

        self.voxel_size_m = self.get_float("voxel_size_m")
        self.filter_point_range = self.get_bool("filter_point_range")
        self.min_point_range_m = self.get_float("min_point_range_m")
        self.max_point_range_m = self.get_float("max_point_range_m")
        self.min_z_m = self.get_float("min_z_m")
        self.max_z_m = self.get_float("max_z_m")
        self.max_points_per_cloud = self.get_int("max_points_per_cloud")
        self.max_voxels = self.get_int("max_voxels")
        self.save_map = self.get_bool("save_map")
        self.save_period_sec = self.get_float("save_period_sec")
        self.save_directory = self.get_string("save_directory")
        self.global_map_filename = self.get_string("global_map_filename")
        self.load_existing_map = self.get_bool("load_existing_map")
        self.scan_after_goal = self.get_bool("scan_after_goal")
        self.scan_after_failed_goal = self.get_bool("scan_after_failed_goal")
        self.scan_angular_speed = self.get_float("scan_angular_speed")
        self.scan_publish_rate = self.get_float("scan_publish_rate")
        self.scan_timeout_padding_sec = self.get_float("scan_timeout_padding_sec")

        self.voxels = set()
        self.last_cloud_msg_time = None
        self.last_save_time = 0.0
        self.last_cloud_log_time = 0.0

        self.active_goal = False
        self.goal_handle = None
        self.goal_start_time = None
        self.current_goal_xy = None
        self.blacklisted_goals = []
        self.exploration_finished = False
        self.scan_active = False
        self.scan_end_time = None
        self.scan_start_yaw = None
        self.scan_last_yaw = None
        self.scan_accumulated_yaw = 0.0
        self.last_scan_log_time = 0.0
        self.funmap_scan_active = False
        self.initial_funmap_scan_done = False

        sensor_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.create_subscription(
            PointCloud2,
            self.pointcloud_topic,
            self.pointcloud_callback,
            sensor_qos,
        )

        map_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.map_cloud_pub = self.create_publisher(
            PointCloud2,
            self.map_cloud_topic,
            map_qos,
        )
        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.scan_cmd_vel_pub = self.create_publisher(
            Twist,
            self.scan_cmd_vel_topic,
            10,
        )

        self.nav_client = ActionClient(self, NavigateToPose, self.nav_action_name)
        self.funmap_head_scan_client = None

        if self.use_funmap:
            self.funmap_head_scan_client = self.create_client(
                Trigger,
                self.funmap_head_scan_service,
            )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_timer(1.0 / self.planning_rate, self.exploration_loop)
        self.create_timer(1.0 / self.cloud_publish_rate, self.publish_voxel_cloud)
        scan_period = 1.0 / max(self.scan_publish_rate, 1.0)
        self.create_timer(scan_period, self.scan_loop)

        if self.save_map:
            os.makedirs(self.save_directory, exist_ok=True)
            self.global_map_path = os.path.join(
                self.save_directory,
                self.global_map_filename,
            )

            if self.load_existing_map:
                self.load_voxel_map()
        else:
            self.global_map_path = None

        self.get_logger().info("Stretch3 3D explorer started.")
        self.get_logger().info(f"Funmap enabled: {self.use_funmap}")
        self.get_logger().info(f"Navigation action: {self.nav_action_name}")
        self.get_logger().info(f"Point cloud topic: {self.pointcloud_topic}")
        self.get_logger().info(f"3D map cloud topic: {self.map_cloud_topic}")
        if self.use_funmap:
            self.get_logger().info(
                f"Funmap head-scan service: {self.funmap_head_scan_service}"
            )
        if self.global_map_path is not None:
            self.get_logger().info(f"Global 3D map file: {self.global_map_path}")
        self.get_logger().info(f"Nav cmd_vel topic: {self.cmd_vel_topic}")
        self.get_logger().info(f"Scan cmd_vel topic: {self.scan_cmd_vel_topic}")
        self.get_logger().info(
            f"Voxel size: {self.voxel_size_m:.3f} m, z window: "
            f"{self.min_z_m:.2f}..{self.max_z_m:.2f} m"
        )

    def get_string(self, name):
        return str(self.get_parameter(name).value)

    def get_float(self, name):
        return float(self.get_parameter(name).value)

    def get_int(self, name):
        return int(self.get_parameter(name).value)

    def get_bool(self, name):
        return bool(self.get_parameter(name).value)

    def pointcloud_callback(self, msg):
        source_frame = msg.header.frame_id
        cloud_is_in_map_frame = source_frame == self.map_frame

        if cloud_is_in_map_frame:
            transform = None
            translation = np.zeros(3, dtype=float)
            rotation_matrix = np.identity(3)
        else:
            transform = self.lookup_transform(self.map_frame, source_frame)

        if (
            not cloud_is_in_map_frame
            and transform is None
            and self.camera_frame_fallback
        ):
            transform = self.lookup_transform(self.map_frame, self.camera_frame_fallback)

        if not cloud_is_in_map_frame and transform is None:
            now = time.time()
            if now - self.last_cloud_log_time > 5.0:
                self.get_logger().warn(
                    f"Waiting for TF from point cloud frame "
                    f"'{source_frame}' to '{self.map_frame}'."
                )
                self.last_cloud_log_time = now
            return

        if transform is not None:
            translation, rotation_matrix = self.transform_to_matrix(transform)

        points = point_cloud2.read_points(
            msg,
            field_names=("x", "y", "z"),
            skip_nans=True,
        )

        added = 0
        stride = self.compute_cloud_stride(msg)

        for index, point in enumerate(points):
            if stride > 1 and index % stride != 0:
                continue

            if len(self.voxels) >= self.max_voxels:
                break

            x, y, z = float(point[0]), float(point[1]), float(point[2])

            if self.filter_point_range:
                range_m = math.sqrt(x * x + y * y + z * z)

                if (
                    range_m < self.min_point_range_m
                    or range_m > self.max_point_range_m
                ):
                    continue

            map_point = rotation_matrix @ np.array([x, y, z]) + translation
            mx, my, mz = map_point.tolist()

            if mz < self.min_z_m or mz > self.max_z_m:
                continue

            voxel = self.point_to_voxel(mx, my, mz)
            self.voxels.add(voxel)
            added += 1

            if len(self.voxels) >= self.max_voxels:
                self.get_logger().warn(
                    f"Voxel map reached max_voxels={self.max_voxels}; "
                    "additional points are being ignored."
                )
                break

        self.last_cloud_msg_time = time.time()

        if added > 0 and self.save_map:
            self.maybe_save_map()

    def compute_cloud_stride(self, msg):
        point_count = int(msg.width * msg.height)

        if point_count <= 0 or point_count <= self.max_points_per_cloud:
            return 1

        return max(1, int(math.ceil(point_count / self.max_points_per_cloud)))

    def point_to_voxel(self, x, y, z):
        inv = 1.0 / self.voxel_size_m
        return (
            int(math.floor(x * inv)),
            int(math.floor(y * inv)),
            int(math.floor(z * inv)),
        )

    def voxel_to_point(self, voxel):
        x, y, z = voxel
        scale = self.voxel_size_m
        return (
            (x + 0.5) * scale,
            (y + 0.5) * scale,
            (z + 0.5) * scale,
        )

    def publish_voxel_cloud(self):
        if not self.voxels:
            return

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.map_frame

        points = [self.voxel_to_point(voxel) for voxel in self.voxels]
        msg = point_cloud2.create_cloud_xyz32(header, points)
        self.map_cloud_pub.publish(msg)

    def maybe_save_map(self):
        now = time.time()

        if now - self.last_save_time < self.save_period_sec:
            return

        self.last_save_time = now
        self.save_voxel_map()

    def save_voxel_map(self):
        if not self.voxels:
            return

        if self.global_map_path is None:
            return

        points = [self.voxel_to_point(voxel) for voxel in sorted(self.voxels)]
        temp_path = f"{self.global_map_path}.tmp"

        with open(temp_path, "w", encoding="utf-8") as handle:
            handle.write("ply\n")
            handle.write("format ascii 1.0\n")
            handle.write(f"element vertex {len(points)}\n")
            handle.write("property float x\n")
            handle.write("property float y\n")
            handle.write("property float z\n")
            handle.write("end_header\n")

            for x, y, z in points:
                handle.write(f"{x:.4f} {y:.4f} {z:.4f}\n")

        os.replace(temp_path, self.global_map_path)

        self.get_logger().info(
            f"Saved global 3D voxel map: {self.global_map_path} "
            f"({len(points)} points)"
        )

    def load_voxel_map(self):
        if self.global_map_path is None:
            return

        if not os.path.exists(self.global_map_path):
            return

        loaded = 0
        in_header = True

        try:
            with open(self.global_map_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()

                    if in_header:
                        if stripped == "end_header":
                            in_header = False
                        continue

                    if not stripped:
                        continue

                    parts = stripped.split()

                    if len(parts) < 3:
                        continue

                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        z = float(parts[2])
                    except ValueError:
                        continue

                    self.voxels.add(self.point_to_voxel(x, y, z))
                    loaded += 1

                    if len(self.voxels) >= self.max_voxels:
                        self.get_logger().warn(
                            f"Loaded map reached max_voxels={self.max_voxels}; "
                            "remaining saved points were ignored."
                        )
                        break

        except OSError as exc:
            self.get_logger().warn(
                f"Could not load global 3D voxel map {self.global_map_path}: {exc}"
            )
            return

        self.get_logger().info(
            f"Loaded global 3D voxel map: {self.global_map_path} "
            f"({loaded} points, {len(self.voxels)} voxels)"
        )

    def exploration_loop(self):
        if self.exploration_finished:
            return

        if self.funmap_scan_active:
            return

        if (
            self.use_funmap
            and self.head_scan_before_first_goal
            and not self.initial_funmap_scan_done
        ):
            self.start_mapping_scan("initial")
            return

        if self.last_cloud_msg_time is None:
            self.get_logger().info(f"Waiting for point cloud: {self.pointcloud_topic}")
            return

        if len(self.voxels) < self.min_3d_voxels_for_navigation:
            self.get_logger().info(
                f"Waiting for 3D map density: {len(self.voxels)} / "
                f"{self.min_3d_voxels_for_navigation} voxels"
            )
            return

        if self.active_goal:
            if self.goal_timed_out():
                self.get_logger().warn("Goal timed out. Canceling and blacklisting.")
                self.blacklist_current_goal()
                self.cancel_current_goal()
            return

        if self.scan_active:
            return

        robot_pose = self.get_robot_pose()

        if robot_pose is None:
            self.get_logger().warn("Could not get robot pose.")
            return

        candidate_goals = self.get_ranked_3d_goals(robot_pose)

        if not candidate_goals:
            self.get_logger().warn("No 3D-safe exploration goals.")
            self.stop_robot()
            return

        self.send_nav_goal(*candidate_goals[0])

    def finish_exploration(self):
        self.cancel_current_goal()
        self.funmap_scan_active = False
        self.stop_scan()
        self.stop_robot()
        self.exploration_finished = True
        self.publish_voxel_cloud()

        if self.save_map:
            self.save_voxel_map()

    def get_ranked_3d_goals(self, robot_pose):
        robot_x, robot_y = robot_pose
        coverage_points = self.get_coverage_points()
        obstacle_points = self.get_navigation_obstacle_points()

        if not coverage_points:
            self.get_logger().warn("3D map has no coverage points.")
            return []

        observed_radius = self.compute_observed_radius(robot_x, robot_y, coverage_points)
        scored = []

        for angle_index in range(max(1, self.candidate_angle_count)):
            yaw = (2.0 * math.pi * angle_index) / float(self.candidate_angle_count)
            direction_x = math.cos(yaw)
            direction_y = math.sin(yaw)

            distance = self.min_goal_distance_m

            while distance <= self.max_goal_distance_m + 1e-6:
                goal_x = robot_x + direction_x * distance
                goal_y = robot_y + direction_y * distance

                if self.is_blacklisted(goal_x, goal_y):
                    distance += self.candidate_distance_step_m
                    continue

                if not self.is_3d_goal_clear(goal_x, goal_y, obstacle_points):
                    distance += self.candidate_distance_step_m
                    continue

                if not self.is_3d_corridor_clear(
                    robot_x,
                    robot_y,
                    goal_x,
                    goal_y,
                    obstacle_points,
                ):
                    distance += self.candidate_distance_step_m
                    continue

                score = self.score_3d_candidate(
                    goal_x,
                    goal_y,
                    distance,
                    observed_radius,
                    coverage_points,
                )
                scored.append((score, goal_x, goal_y, distance))

                distance += self.candidate_distance_step_m

        scored.sort(reverse=True, key=lambda item: item[0])

        for index, candidate in enumerate(scored[: self.max_candidates_to_test]):
            _, x, y, distance = candidate
            self.get_logger().info(
                f"3D explore candidate #{index + 1}: x={x:.2f}, y={y:.2f}, "
                f"dist={distance:.2f}"
            )

        return [
            (x, y)
            for _, x, y, _ in scored[: self.max_candidates_to_test]
        ]

    def get_coverage_points(self):
        return [
            self.voxel_to_point(voxel)[:2]
            for voxel in self.voxels
        ]

    def get_navigation_obstacle_points(self):
        points = []

        for voxel in self.voxels:
            x, y, z = self.voxel_to_point(voxel)

            if z < self.obstacle_min_z_m or z > self.obstacle_max_z_m:
                continue

            points.append((x, y))

        return points

    def compute_observed_radius(self, robot_x, robot_y, occupied_points):
        if not occupied_points:
            return self.ideal_goal_distance_m

        distances = [
            math.hypot(x - robot_x, y - robot_y)
            for x, y in occupied_points
        ]
        distances.sort()
        percentile_index = int(0.85 * (len(distances) - 1))

        return distances[percentile_index]

    def is_3d_goal_clear(self, goal_x, goal_y, occupied_points):
        clearance_sq = self.robot_clearance_radius_m ** 2

        for point_x, point_y in occupied_points:
            if (point_x - goal_x) ** 2 + (point_y - goal_y) ** 2 < clearance_sq:
                return False

        return True

    def is_3d_corridor_clear(self, start_x, start_y, goal_x, goal_y, occupied_points):
        dx = goal_x - start_x
        dy = goal_y - start_y
        length_sq = dx * dx + dy * dy

        if length_sq < 1e-6:
            return False

        clearance_sq = self.corridor_clearance_radius_m ** 2

        for point_x, point_y in occupied_points:
            rel_x = point_x - start_x
            rel_y = point_y - start_y
            projection = (rel_x * dx + rel_y * dy) / length_sq

            if projection <= 0.05 or projection >= 0.95:
                continue

            closest_x = start_x + projection * dx
            closest_y = start_y + projection * dy

            if (
                (point_x - closest_x) ** 2
                + (point_y - closest_y) ** 2
                < clearance_sq
            ):
                return False

        return True

    def score_3d_candidate(
        self,
        goal_x,
        goal_y,
        distance,
        observed_radius,
        occupied_points,
    ):
        distance_error = abs(distance - self.ideal_goal_distance_m)
        distance_score = max(
            0.0,
            1.0 - distance_error / max(self.ideal_goal_distance_m, 0.01),
        )

        frontier_target = observed_radius + self.frontier_band_width_m
        frontier_error = abs(distance - frontier_target)
        frontier_score = max(
            0.0,
            1.0 - frontier_error / max(self.frontier_band_width_m, 0.01),
        )

        sparse_count = 0
        sparse_radius_sq = self.frontier_sparse_radius_m ** 2

        for point_x, point_y in occupied_points:
            if (point_x - goal_x) ** 2 + (point_y - goal_y) ** 2 < sparse_radius_sq:
                sparse_count += 1

        sparse_penalty = min(1.0, sparse_count / 20.0)

        return 1.25 * distance_score + 1.0 * frontier_score - sparse_penalty

    def send_nav_goal(self, x, y):
        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error(
                f"NavigateToPose action server not available: {self.nav_action_name}"
            )
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = self.make_goal_pose(x, y)

        self.get_logger().info(f"Sending 3D exploration goal: x={x:.2f}, y={y:.2f}")

        self.active_goal = True
        self.goal_start_time = time.time()
        self.current_goal_xy = (x, y)

        future = self.nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self.nav_feedback_callback,
        )
        future.add_done_callback(self.goal_response_callback)

    def make_goal_pose(self, x, y):
        pose = PoseStamped()
        pose.header.frame_id = self.map_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = 0.0

        robot_pose = self.get_robot_pose()

        if robot_pose is None:
            yaw = 0.0
        else:
            yaw = math.atan2(y - robot_pose[1], x - robot_pose[0])

        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)

        return pose

    def goal_response_callback(self, future):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"NavigateToPose response failed: {exc}")
            self.active_goal = False
            self.goal_handle = None
            return

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn("NavigateToPose goal rejected.")
            self.blacklist_current_goal()
            self.active_goal = False
            self.goal_handle = None

            if self.scan_after_failed_goal and not self.exploration_finished:
                self.start_mapping_scan("goal rejected")

            return

        self.goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(self.goal_result_callback)

    def nav_feedback_callback(self, feedback_msg):
        try:
            distance = feedback_msg.feedback.distance_remaining
            self.get_logger().debug(f"Distance remaining: {distance:.2f} m")
        except Exception:
            pass

    def goal_result_callback(self, future):
        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().error(f"NavigateToPose result failed: {exc}")
            self.active_goal = False
            self.goal_handle = None
            return

        if result.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Exploration goal succeeded.")
            self.active_goal = False
            self.goal_handle = None
            self.goal_start_time = None

            if self.scan_after_goal and not self.exploration_finished:
                self.start_mapping_scan("goal succeeded")
        else:
            self.get_logger().warn(f"Exploration goal failed. Status={result.status}")
            self.blacklist_current_goal()
            self.active_goal = False
            self.goal_handle = None
            self.goal_start_time = None

            if self.scan_after_failed_goal and not self.exploration_finished:
                self.start_mapping_scan("goal failed")

    def get_robot_pose(self):
        for frame in self.unique_frames(
            [self.robot_base_frame, "base_link", "base_footprint"]
        ):
            transform = self.lookup_transform(self.map_frame, frame, timeout_sec=0.5)

            if transform is not None:
                return (
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                )

        return None

    def get_robot_yaw(self):
        for frame in self.unique_frames(
            [self.robot_base_frame, "base_link", "base_footprint"]
        ):
            transform = self.lookup_transform(self.map_frame, frame, timeout_sec=0.2)

            if transform is not None:
                rotation = transform.transform.rotation
                return self.quaternion_to_yaw(
                    rotation.x,
                    rotation.y,
                    rotation.z,
                    rotation.w,
                )

        return None

    def quaternion_to_yaw(self, x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def angle_diff(self, target, source):
        return math.atan2(
            math.sin(target - source),
            math.cos(target - source),
        )

    def lookup_transform(self, target_frame, source_frame, timeout_sec=0.2):
        if not source_frame:
            return None

        try:
            return self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=timeout_sec),
            )
        except Exception:
            return None

    def transform_to_matrix(self, transform):
        translation_msg = transform.transform.translation
        rotation_msg = transform.transform.rotation
        translation = np.array(
            [translation_msg.x, translation_msg.y, translation_msg.z],
            dtype=float,
        )
        rotation_matrix = self.quaternion_to_rotation_matrix(
            rotation_msg.x,
            rotation_msg.y,
            rotation_msg.z,
            rotation_msg.w,
        )

        return translation, rotation_matrix

    def quaternion_to_rotation_matrix(self, x, y, z, w):
        norm = math.sqrt(x * x + y * y + z * z + w * w)

        if norm < 1e-9:
            return np.identity(3)

        x /= norm
        y /= norm
        z /= norm
        w /= norm

        return np.array(
            [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - z * w),
                    2.0 * (x * z + y * w),
                ],
                [
                    2.0 * (x * y + z * w),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - x * w),
                ],
                [
                    2.0 * (x * z - y * w),
                    2.0 * (y * z + x * w),
                    1.0 - 2.0 * (x * x + y * y),
                ],
            ],
            dtype=float,
        )

    def blacklist_current_goal(self):
        if self.current_goal_xy is not None:
            self.blacklisted_goals.append(self.current_goal_xy)

    def is_blacklisted(self, x, y):
        return any(
            math.hypot(x - bx, y - by) < self.blacklist_radius_m
            for bx, by in self.blacklisted_goals
        )

    def cancel_current_goal(self):
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()

        self.active_goal = False
        self.goal_handle = None
        self.goal_start_time = None

    def goal_timed_out(self):
        return (
            self.goal_start_time is not None
            and time.time() - self.goal_start_time > self.goal_timeout_sec
        )

    def stop_robot(self):
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
        self.scan_cmd_vel_pub.publish(twist)

    def start_mapping_scan(self, reason):
        if self.use_funmap and self.funmap_head_scan_client is not None:
            self.start_funmap_head_scan(reason)
            return

        self.start_scan()

    def start_funmap_head_scan(self, reason):
        if self.funmap_scan_active:
            return

        if self.scan_active:
            self.stop_scan()

        self.cancel_current_goal()
        self.stop_robot()

        service_ready = self.funmap_head_scan_client.wait_for_service(
            timeout_sec=self.funmap_service_timeout_sec,
        )

        if not service_ready:
            self.get_logger().warn(
                f"Funmap head-scan service is not available: "
                f"{self.funmap_head_scan_service}. Falling back to cmd_vel scan."
            )
            if reason == "initial":
                self.initial_funmap_scan_done = True
            self.start_scan()
            return

        self.get_logger().info(f"Requesting funmap head scan ({reason}).")
        self.funmap_scan_active = True
        future = self.funmap_head_scan_client.call_async(Trigger.Request())
        future.add_done_callback(
            lambda done_future: self.funmap_head_scan_callback(done_future, reason)
        )

    def funmap_head_scan_callback(self, future, reason):
        self.funmap_scan_active = False

        if reason == "initial":
            self.initial_funmap_scan_done = True

        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().warn(f"Funmap head scan failed: {exc}")
            return

        if response.success:
            self.get_logger().info(f"Funmap head scan complete: {response.message}")
        else:
            self.get_logger().warn(
                f"Funmap head scan reported failure: {response.message}"
            )

    def start_scan(self):
        if abs(self.scan_angular_speed) < 1e-6:
            self.get_logger().warn("scan_angular_speed is zero; skipping scan.")
            return

        self.cancel_current_goal()
        self.stop_robot()

        duration = 2.0 * math.pi / abs(self.scan_angular_speed)
        yaw = self.get_robot_yaw()

        self.scan_active = True
        self.scan_end_time = time.time() + duration + self.scan_timeout_padding_sec
        self.scan_start_yaw = yaw
        self.scan_last_yaw = yaw
        self.scan_accumulated_yaw = 0.0
        self.last_scan_log_time = 0.0

        if yaw is None:
            self.get_logger().warn(
                "Starting timed 360-degree scan; robot yaw TF is unavailable."
            )
        else:
            self.get_logger().info(
                f"Starting TF-tracked 360-degree scan at yaw={yaw:.2f} rad."
            )

    def scan_loop(self):
        if not self.scan_active:
            return

        yaw = self.get_robot_yaw()

        if yaw is not None and self.scan_last_yaw is None:
            self.scan_last_yaw = yaw

        if yaw is not None and self.scan_last_yaw is not None:
            delta = self.angle_diff(yaw, self.scan_last_yaw)
            self.scan_accumulated_yaw += abs(delta)
            self.scan_last_yaw = yaw

        if self.scan_accumulated_yaw >= 2.0 * math.pi:
            self.get_logger().info(
                f"Finished 360-degree scan by TF: "
                f"{self.scan_accumulated_yaw:.2f} rad."
            )
            self.stop_scan()
            return

        if self.scan_end_time is None or time.time() >= self.scan_end_time:
            self.get_logger().warn(
                f"Finished 360-degree scan by timeout; TF yaw accumulation="
                f"{self.scan_accumulated_yaw:.2f} rad."
            )
            self.stop_scan()
            return

        twist = Twist()
        twist.angular.z = self.scan_angular_speed
        self.scan_cmd_vel_pub.publish(twist)

        now = time.time()
        if now - self.last_scan_log_time > 1.0:
            self.get_logger().info(
                f"Scanning: cmd_vel.angular.z={self.scan_angular_speed:.2f}, "
                f"accumulated_yaw={self.scan_accumulated_yaw:.2f} rad"
            )
            self.last_scan_log_time = now

    def stop_scan(self):
        self.scan_active = False
        self.scan_end_time = None
        self.scan_start_yaw = None
        self.scan_last_yaw = None
        self.scan_accumulated_yaw = 0.0
        self.last_scan_log_time = 0.0
        self.stop_robot()

    def unique_frames(self, frames):
        unique = []

        for frame in frames:
            if frame and frame not in unique:
                unique.append(frame)

        return unique


def main(args=None):
    rclpy.init(args=args)
    node = ExploreNode3D()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("3D explorer interrupted by user.")
    finally:
        node.finish_exploration()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
