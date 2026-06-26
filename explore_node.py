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
from nav2_msgs.action import NavigateToPose, ComputePathToPose

import tf2_ros


class ExploreNode(Node):
    def __init__(self):
        super().__init__("explore_node")

        # ------------------------------------------------------------
        # Core exploration parameters
        # ------------------------------------------------------------
        self.declare_parameter("exploration_threshold", 0.97)
        self.declare_parameter("frontier_min_size", 8)
        self.declare_parameter("planning_rate", 1.0)
        self.declare_parameter("goal_timeout_sec", 90.0)

        # ------------------------------------------------------------
        # Balanced local goal distances
        #
        # These are intentionally modest for Stretch live-SLAM exploration.
        # The robot should progress through the map in small, reliable moves
        # instead of aggressively picking distant frontiers.
        # ------------------------------------------------------------
        self.declare_parameter("min_goal_distance_m", 0.30)
        self.declare_parameter("ideal_goal_distance_m", 0.80)
        self.declare_parameter("max_goal_distance_m", 1.40)

        # ------------------------------------------------------------
        # Candidate scoring
        #
        # size_weight:
        #   Small bonus for larger frontiers.
        #
        # distance_weight:
        #   Rewards candidates near ideal_goal_distance_m.
        #
        # far_goal_penalty_weight:
        #   Penalizes goals beyond the ideal distance.
        #
        # cost_penalty_weight:
        #   Penalizes goals near inflated costmap cost.
        # ------------------------------------------------------------
        self.declare_parameter("size_weight", 0.50)
        self.declare_parameter("distance_weight", 1.50)
        self.declare_parameter("far_goal_penalty_weight", 2.00)
        self.declare_parameter("cost_penalty_weight", 1.25)
        self.declare_parameter("max_candidates_to_test", 20)

        # ------------------------------------------------------------
        # Frontier / goal safety
        #
        # These are balanced values:
        # - not so strict that frontier goals are impossible
        # - not so loose that the robot hugs obstacles aggressively
        # ------------------------------------------------------------
        self.declare_parameter("frontier_clearance_m", 0.18)
        self.declare_parameter("goal_inward_offset_m", 0.40)
        self.declare_parameter("goal_search_radius_m", 0.55)

        self.declare_parameter("goal_obstacle_clearance_m", 0.30)
        self.declare_parameter("goal_unknown_clearance_m", 0.15)

        # Allow some unknown around candidate goals because frontier goals
        # are naturally near unknown space.
        self.declare_parameter("raw_unknown_ratio_limit", 0.50)
        self.declare_parameter("costmap_unknown_ratio_limit", 0.45)

        # ------------------------------------------------------------
        # Global costmap safety
        # ------------------------------------------------------------
        self.declare_parameter("costmap_clearance_m", 0.20)
        self.declare_parameter("max_center_cost", 35)
        self.declare_parameter("max_patch_cost", 85)
        self.declare_parameter("lethal_cost", 99)

        # The goal center is always required to be known.
        # This parameter only affects whether unknown cells are allowed
        # inside the nearby costmap patch.
        self.declare_parameter("reject_unknown_costmap_patch", False)

        # ------------------------------------------------------------
        # Planner validation
        # ------------------------------------------------------------
        self.declare_parameter("validate_with_planner", True)
        self.declare_parameter("max_planned_path_length_m", 2.20)

        # ------------------------------------------------------------
        # ROS topics/actions
        # ------------------------------------------------------------
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("global_costmap_topic", "/global_costmap/costmap")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")

        self.declare_parameter("nav_action_name", "navigate_to_pose")
        self.declare_parameter("planner_action_name", "compute_path_to_pose")

        # Your Stretch Nav2 config showed base_link.
        # The code also tries base_footprint as a fallback.
        self.declare_parameter("robot_base_frame", "base_link")
        self.declare_parameter("map_frame", "map")

        # ------------------------------------------------------------
        # Blacklist failed goals
        # ------------------------------------------------------------
        self.declare_parameter("blacklist_radius_m", 0.45)

        # ------------------------------------------------------------
        # Optional scan behavior
        # Disabled by default. Nav2 recovery spin was already causing
        # issues in your setup, so the explorer should not add more spin
        # unless you explicitly enable it.
        # ------------------------------------------------------------
        self.declare_parameter("scan_after_goal", False)
        self.declare_parameter("scan_angular_speed", 0.45)

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

        self.min_goal_distance_m = float(
            self.get_parameter("min_goal_distance_m").value
        )
        self.ideal_goal_distance_m = float(
            self.get_parameter("ideal_goal_distance_m").value
        )
        self.max_goal_distance_m = float(
            self.get_parameter("max_goal_distance_m").value
        )

        self.size_weight = float(
            self.get_parameter("size_weight").value
        )
        self.distance_weight = float(
            self.get_parameter("distance_weight").value
        )
        self.far_goal_penalty_weight = float(
            self.get_parameter("far_goal_penalty_weight").value
        )
        self.cost_penalty_weight = float(
            self.get_parameter("cost_penalty_weight").value
        )
        self.max_candidates_to_test = int(
            self.get_parameter("max_candidates_to_test").value
        )

        self.frontier_clearance_m = float(
            self.get_parameter("frontier_clearance_m").value
        )
        self.goal_inward_offset_m = float(
            self.get_parameter("goal_inward_offset_m").value
        )
        self.goal_search_radius_m = float(
            self.get_parameter("goal_search_radius_m").value
        )
        self.goal_obstacle_clearance_m = float(
            self.get_parameter("goal_obstacle_clearance_m").value
        )
        self.goal_unknown_clearance_m = float(
            self.get_parameter("goal_unknown_clearance_m").value
        )

        self.raw_unknown_ratio_limit = float(
            self.get_parameter("raw_unknown_ratio_limit").value
        )
        self.costmap_unknown_ratio_limit = float(
            self.get_parameter("costmap_unknown_ratio_limit").value
        )

        self.costmap_clearance_m = float(
            self.get_parameter("costmap_clearance_m").value
        )
        self.max_center_cost = int(
            self.get_parameter("max_center_cost").value
        )
        self.max_patch_cost = int(
            self.get_parameter("max_patch_cost").value
        )
        self.lethal_cost = int(
            self.get_parameter("lethal_cost").value
        )
        self.reject_unknown_costmap_patch = bool(
            self.get_parameter("reject_unknown_costmap_patch").value
        )

        self.validate_with_planner = bool(
            self.get_parameter("validate_with_planner").value
        )
        self.max_planned_path_length_m = float(
            self.get_parameter("max_planned_path_length_m").value
        )

        self.map_topic = str(
            self.get_parameter("map_topic").value
        )
        self.global_costmap_topic = str(
            self.get_parameter("global_costmap_topic").value
        )
        self.cmd_vel_topic = str(
            self.get_parameter("cmd_vel_topic").value
        )

        self.nav_action_name = str(
            self.get_parameter("nav_action_name").value
        )
        self.planner_action_name = str(
            self.get_parameter("planner_action_name").value
        )

        self.robot_base_frame = str(
            self.get_parameter("robot_base_frame").value
        )
        self.map_frame = str(
            self.get_parameter("map_frame").value
        )

        self.blacklist_radius_m = float(
            self.get_parameter("blacklist_radius_m").value
        )

        self.scan_after_goal = bool(
            self.get_parameter("scan_after_goal").value
        )
        self.scan_angular_speed = float(
            self.get_parameter("scan_angular_speed").value
        )

        # ------------------------------------------------------------
        # Internal state
        # ------------------------------------------------------------
        self.latest_map = None
        self.map_array = None

        self.latest_global_costmap = None
        self.global_costmap_array = None
        self.current_costmap_reachable_mask = None

        self.active_goal = False
        self.testing_plan = False

        self.goal_handle = None
        self.goal_start_time = None
        self.current_goal_xy = None

        self.pending_candidate_goals = []
        self.current_test_goal_xy = None

        self.blacklisted_goals = []
        self.exploration_finished = False

        # ------------------------------------------------------------
        # QoS
        #
        # On your Stretch output, both /map and /global_costmap/costmap
        # are RELIABLE + TRANSIENT_LOCAL.
        # ------------------------------------------------------------
        transient_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            transient_qos,
        )

        self.global_costmap_sub = self.create_subscription(
            OccupancyGrid,
            self.global_costmap_topic,
            self.global_costmap_callback,
            transient_qos,
        )

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            self.cmd_vel_topic,
            10,
        )

        self.nav_client = ActionClient(
            self,
            NavigateToPose,
            self.nav_action_name,
        )

        self.plan_client = ActionClient(
            self,
            ComputePathToPose,
            self.planner_action_name,
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self,
        )

        self.timer = self.create_timer(
            1.0 / self.planning_rate,
            self.exploration_loop,
        )

        self.get_logger().info("Balanced Stretch 2D explorer started.")
        self.get_logger().info(f"Map topic: {self.map_topic}")
        self.get_logger().info(f"Global costmap topic: {self.global_costmap_topic}")
        self.get_logger().info(f"Nav action: {self.nav_action_name}")
        self.get_logger().info(f"Planner action: {self.planner_action_name}")
        self.get_logger().info(f"Cmd vel topic: {self.cmd_vel_topic}")
        self.get_logger().info(
            f"Goal distance window: min={self.min_goal_distance_m:.2f}, "
            f"ideal={self.ideal_goal_distance_m:.2f}, "
            f"max={self.max_goal_distance_m:.2f}"
        )
        self.get_logger().info(
            f"Safety: obstacle_clearance={self.goal_obstacle_clearance_m:.2f}, "
            f"unknown_clearance={self.goal_unknown_clearance_m:.2f}, "
            f"raw_unknown_ratio_limit={self.raw_unknown_ratio_limit:.2f}, "
            f"costmap_clearance={self.costmap_clearance_m:.2f}, "
            f"costmap_unknown_ratio_limit={self.costmap_unknown_ratio_limit:.2f}, "
            f"max_center_cost={self.max_center_cost}, "
            f"max_patch_cost={self.max_patch_cost}"
        )

    # ------------------------------------------------------------
    # Map callbacks
    # ------------------------------------------------------------
    def map_callback(self, msg):
        self.latest_map = msg

        width = msg.info.width
        height = msg.info.height

        data = np.array(msg.data, dtype=np.int16)
        self.map_array = data.reshape((height, width))

    def global_costmap_callback(self, msg):
        self.latest_global_costmap = msg

        width = msg.info.width
        height = msg.info.height

        data = np.array(msg.data, dtype=np.int16)
        self.global_costmap_array = data.reshape((height, width))

    # ------------------------------------------------------------
    # Main exploration loop
    # ------------------------------------------------------------
    def exploration_loop(self):
        if self.exploration_finished:
            return

        if self.latest_map is None or self.map_array is None:
            self.get_logger().info("Waiting for /map...")
            return

        if self.latest_global_costmap is None or self.global_costmap_array is None:
            self.get_logger().info("Waiting for /global_costmap/costmap...")
            return

        if self.active_goal:
            if self.goal_timed_out():
                self.get_logger().warn(
                    "Current goal timed out. Canceling and blacklisting."
                )
                self.blacklist_current_goal()
                self.cancel_current_goal()
            return

        if self.testing_plan:
            return

        exploration = self.compute_exploration_coefficient()
        unknown_count = int(np.count_nonzero(self.map_array == -1))

        self.get_logger().info(
            f"Exploration coefficient: {exploration:.3f}, "
            f"unknown cells: {unknown_count}"
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

        robot_pose = self.get_robot_pose()

        if robot_pose is None:
            self.get_logger().warn("Could not get robot pose.")
            return

        frontiers = self.find_frontiers()

        self.get_logger().info(f"Detected {len(frontiers)} frontier cells.")

        if not frontiers:
            self.get_logger().warn("No frontiers found.")
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
            self.get_logger().warn("No valid frontier clusters after filtering.")
            self.stop_robot()
            return

        candidate_goals = self.get_ranked_frontier_goals(
            valid_clusters,
            robot_pose,
        )

        if not candidate_goals:
            self.get_logger().warn("No balanced frontier candidates found.")
            self.stop_robot()
            return

        if self.validate_with_planner:
            self.start_planner_validation(candidate_goals)
        else:
            x, y = candidate_goals[0]
            self.send_nav_goal(x, y)

    # ------------------------------------------------------------
    # Exploration coefficient
    # ------------------------------------------------------------
    def compute_exploration_coefficient(self):
        unknown = np.count_nonzero(self.map_array == -1)
        known = np.count_nonzero(self.map_array != -1)

        total = known + unknown

        if total == 0:
            return 0.0

        return float(known) / float(total)

    # ------------------------------------------------------------
    # Frontier detection
    # ------------------------------------------------------------
    def find_frontiers(self):
        frontiers = []

        height, width = self.map_array.shape
        resolution = self.latest_map.info.resolution

        clearance_cells = max(
            1,
            int(self.frontier_clearance_m / resolution),
        )

        for y in range(1, height - 1):
            for x in range(1, width - 1):

                if self.map_array[y, x] != 0:
                    continue

                if not self.has_raw_map_obstacle_clearance(
                    x,
                    y,
                    clearance_cells,
                ):
                    continue

                neighborhood = self.map_array[y - 1:y + 2, x - 1:x + 2]

                if np.any(neighborhood == -1):
                    frontiers.append((x, y))

        return frontiers

    def has_raw_map_obstacle_clearance(self, x, y, clearance_cells):
        height, width = self.map_array.shape

        x_min = max(0, x - clearance_cells)
        x_max = min(width, x + clearance_cells + 1)

        y_min = max(0, y - clearance_cells)
        y_max = min(height, y + clearance_cells + 1)

        patch = self.map_array[y_min:y_max, x_min:x_max]

        if np.any(patch > 50):
            return False

        return True

    # ------------------------------------------------------------
    # Frontier clustering
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
    # Rank frontier goals
    # ------------------------------------------------------------
    def get_ranked_frontier_goals(self, clusters, robot_pose):
        robot_x, robot_y = robot_pose

        robot_cell = self.world_to_map(robot_x, robot_y)

        if robot_cell is None:
            self.get_logger().warn("Robot is outside current SLAM map.")
            return []

        slam_reachable_mask = self.compute_reachable_free_mask(robot_cell)

        self.current_costmap_reachable_mask = self.compute_reachable_global_costmap_mask(
            robot_x,
            robot_y,
        )

        scored_candidates = []

        for cluster in clusters:
            candidate_cells = self.find_safe_goal_cells_for_cluster(
                cluster,
                robot_cell,
                slam_reachable_mask,
            )

            for cell in candidate_cells:
                world_point = self.map_to_world(cell[0], cell[1])

                if world_point is None:
                    continue

                goal_x, goal_y = world_point

                if self.is_blacklisted(goal_x, goal_y):
                    continue

                distance = math.sqrt(
                    (goal_x - robot_x) ** 2 +
                    (goal_y - robot_y) ** 2
                )

                if distance < self.min_goal_distance_m:
                    continue

                if distance > self.max_goal_distance_m:
                    continue

                metrics = self.get_costmap_metrics(goal_x, goal_y)

                if metrics is None:
                    continue

                center_cost, patch_max, unknown_ratio = metrics

                score = self.score_candidate(
                    cluster_size=len(cluster),
                    distance=distance,
                    center_cost=center_cost,
                    patch_max=patch_max,
                    unknown_ratio=unknown_ratio,
                )

                scored_candidates.append(
                    {
                        "score": score,
                        "x": goal_x,
                        "y": goal_y,
                        "distance": distance,
                        "cluster_size": len(cluster),
                        "center_cost": center_cost,
                        "patch_max": patch_max,
                        "unknown_ratio": unknown_ratio,
                    }
                )

        scored_candidates.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        if not scored_candidates:
            return []

        self.get_logger().info("Top balanced candidates:")

        for i, candidate in enumerate(scored_candidates[: self.max_candidates_to_test]):
            self.get_logger().info(
                f"  #{i + 1}: "
                f"x={candidate['x']:.2f}, y={candidate['y']:.2f}, "
                f"dist={candidate['distance']:.2f}, "
                f"score={candidate['score']:.2f}, "
                f"center_cost={candidate['center_cost']}, "
                f"patch_max={candidate['patch_max']}, "
                f"unknown_ratio={candidate['unknown_ratio']:.2f}, "
                f"cluster={candidate['cluster_size']}"
            )

        return [
            (candidate["x"], candidate["y"])
            for candidate in scored_candidates[: self.max_candidates_to_test]
        ]

    def score_candidate(
        self,
        cluster_size,
        distance,
        center_cost,
        patch_max,
        unknown_ratio,
    ):
        # Small bonus for larger frontiers, but capped.
        size_score = self.size_weight * min(
            1.0,
            math.sqrt(cluster_size) / 12.0,
        )

        # Prefer goals near the ideal distance.
        distance_error = abs(distance - self.ideal_goal_distance_m)

        distance_score = self.distance_weight * max(
            0.0,
            1.0 - distance_error / max(self.ideal_goal_distance_m, 0.01),
        )

        # Strongly penalize goals beyond the ideal distance.
        far_penalty = self.far_goal_penalty_weight * max(
            0.0,
            distance - self.ideal_goal_distance_m,
        ) ** 2

        # Penalize inflated/near-obstacle cost.
        normalized_cost = (float(center_cost) + float(patch_max)) / 200.0
        cost_penalty = self.cost_penalty_weight * normalized_cost

        # Mild penalty for a very unknown-heavy patch.
        unknown_penalty = 0.50 * unknown_ratio

        return (
            size_score
            + distance_score
            - far_penalty
            - cost_penalty
            - unknown_penalty
        )

    # ------------------------------------------------------------
    # Find safe cells for one frontier cluster
    # ------------------------------------------------------------
    def find_safe_goal_cells_for_cluster(
        self,
        cluster,
        robot_cell,
        slam_reachable_mask,
    ):
        resolution = self.latest_map.info.resolution

        inward_offset_cells = max(
            1,
            int(self.goal_inward_offset_m / resolution),
        )

        search_radius_cells = max(
            1,
            int(self.goal_search_radius_m / resolution),
        )

        obstacle_clearance_cells = max(
            1,
            int(self.goal_obstacle_clearance_m / resolution),
        )

        unknown_clearance_cells = max(
            1,
            int(self.goal_unknown_clearance_m / resolution),
        )

        robot_mx, robot_my = robot_cell

        centroid = self.compute_cluster_centroid(cluster)
        cx, cy = centroid

        sorted_frontier_cells = sorted(
            cluster,
            key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2,
        )

        scored_cells = []
        seen = set()

        for fx, fy in sorted_frontier_cells:
            dx = robot_mx - fx
            dy = robot_my - fy

            norm = math.sqrt(dx * dx + dy * dy)

            if norm < 1e-6:
                continue

            ux = dx / norm
            uy = dy / norm

            # Move inward from the frontier toward currently known space.
            base_x = int(round(fx + ux * inward_offset_cells))
            base_y = int(round(fy + uy * inward_offset_cells))

            for sy in range(
                base_y - search_radius_cells,
                base_y + search_radius_cells + 1,
            ):
                for sx in range(
                    base_x - search_radius_cells,
                    base_x + search_radius_cells + 1,
                ):

                    if (sx, sy) in seen:
                        continue

                    seen.add((sx, sy))

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
                        obstacle_clearance_cells,
                        unknown_clearance_cells,
                        slam_reachable_mask,
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

                    # Prefer cells close to the frontier but not trivial.
                    local_score = -frontier_dist + 0.03 * robot_dist

                    scored_cells.append((local_score, sx, sy))

        scored_cells.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        # Keep several candidates per cluster so planner validation has options.
        return [
            (sx, sy)
            for _, sx, sy in scored_cells[:3]
        ]

    # ------------------------------------------------------------
    # Candidate cell safety in raw SLAM map + global costmap
    # ------------------------------------------------------------
    def is_safe_goal_cell(
        self,
        mx,
        my,
        obstacle_clearance_cells,
        unknown_clearance_cells,
        slam_reachable_mask,
    ):
        if not self.is_cell_in_bounds(mx, my):
            return False

        if self.map_array[my, mx] != 0:
            return False

        if not slam_reachable_mask[my, mx]:
            return False

        height, width = self.map_array.shape

        # Raw map obstacle clearance.
        x_min = max(0, mx - obstacle_clearance_cells)
        x_max = min(width, mx + obstacle_clearance_cells + 1)

        y_min = max(0, my - obstacle_clearance_cells)
        y_max = min(height, my + obstacle_clearance_cells + 1)

        obstacle_patch = self.map_array[y_min:y_max, x_min:x_max]

        if np.any(obstacle_patch > 50):
            return False

        # Raw map unknown check.
        #
        # Do not reject just because there is some unknown near a frontier.
        # Reject only if the nearby patch is mostly unknown.
        ux_min = max(0, mx - unknown_clearance_cells)
        ux_max = min(width, mx + unknown_clearance_cells + 1)

        uy_min = max(0, my - unknown_clearance_cells)
        uy_max = min(height, my + unknown_clearance_cells + 1)

        unknown_patch = self.map_array[uy_min:uy_max, ux_min:ux_max]

        if unknown_patch.size == 0:
            return False

        unknown_count = np.count_nonzero(unknown_patch == -1)
        unknown_ratio = float(unknown_count) / float(unknown_patch.size)

        if unknown_ratio > self.raw_unknown_ratio_limit:
            return False

        world_point = self.map_to_world(mx, my)

        if world_point is None:
            return False

        wx, wy = world_point

        if not self.is_world_safe_in_global_costmap(wx, wy):
            return False

        return True

    # ------------------------------------------------------------
    # Global costmap safety
    # ------------------------------------------------------------
    def get_costmap_metrics(self, wx, wy):
        costmap_cell = self.world_to_global_costmap(wx, wy)

        if costmap_cell is None:
            return None

        mx, my = costmap_cell

        resolution = self.latest_global_costmap.info.resolution

        clearance_cells = max(
            1,
            int(self.costmap_clearance_m / resolution),
        )

        height, width = self.global_costmap_array.shape

        x_min = max(0, mx - clearance_cells)
        x_max = min(width, mx + clearance_cells + 1)

        y_min = max(0, my - clearance_cells)
        y_max = min(height, my + clearance_cells + 1)

        patch = self.global_costmap_array[y_min:y_max, x_min:x_max]

        if patch.size == 0:
            return None

        center_cost = int(self.global_costmap_array[my, mx])
        patch_max = int(np.max(patch))

        unknown_count = np.count_nonzero(patch < 0)
        unknown_ratio = float(unknown_count) / float(patch.size)

        return center_cost, patch_max, unknown_ratio

    def is_world_safe_in_global_costmap(self, wx, wy):
        costmap_cell = self.world_to_global_costmap(wx, wy)

        if costmap_cell is None:
            return False

        mx, my = costmap_cell

        # The goal must be in the same currently reachable global-costmap
        # component as the robot.
        if self.current_costmap_reachable_mask is not None:
            if not self.current_costmap_reachable_mask[my, mx]:
                return False

        metrics = self.get_costmap_metrics(wx, wy)

        if metrics is None:
            return False

        center_cost, patch_max, unknown_ratio = metrics

        # The goal center must be known. This avoids commanding the robot
        # directly into unknown space.
        if center_cost < 0:
            return False

        # The goal center must not be too close to inflated obstacle cost.
        if center_cost > self.max_center_cost:
            return False

        resolution = self.latest_global_costmap.info.resolution

        clearance_cells = max(
            1,
            int(self.costmap_clearance_m / resolution),
        )

        height, width = self.global_costmap_array.shape

        x_min = max(0, mx - clearance_cells)
        x_max = min(width, mx + clearance_cells + 1)

        y_min = max(0, my - clearance_cells)
        y_max = min(height, my + clearance_cells + 1)

        patch = self.global_costmap_array[y_min:y_max, x_min:x_max]

        if patch.size == 0:
            return False

        # Hard reject only lethal or near-lethal obstacles.
        if np.any(patch >= self.lethal_cost):
            return False

        # Optional strict mode. Usually false for exploration.
        if self.reject_unknown_costmap_patch and np.any(patch < 0):
            return False

        # Balanced mode: allow some unknown near frontiers, but not mostly unknown.
        if unknown_ratio > self.costmap_unknown_ratio_limit:
            return False

        if patch_max > self.max_patch_cost:
            return False

        return True

    def is_traversable_costmap_cell(self, mx, my):
        if self.global_costmap_array is None:
            return False

        height, width = self.global_costmap_array.shape

        if mx < 0 or my < 0 or mx >= width or my >= height:
            return False

        cost = int(self.global_costmap_array[my, mx])

        # For reachability, do not flood-fill through unknown.
        # Frontier goals may be near unknown, but the path to them should
        # go through known costmap space.
        if cost < 0:
            return False

        if cost >= self.lethal_cost:
            return False

        return True

    def compute_reachable_global_costmap_mask(self, robot_x, robot_y):
        start_cell = self.world_to_global_costmap(robot_x, robot_y)

        if start_cell is None:
            self.get_logger().warn("Robot is outside global costmap.")
            return None

        height, width = self.global_costmap_array.shape

        reachable = np.zeros((height, width), dtype=bool)

        start_x, start_y = start_cell

        if not self.is_traversable_costmap_cell(start_x, start_y):
            nearest = self.find_nearest_traversable_costmap_cell(
                start_x,
                start_y,
            )

            if nearest is None:
                self.get_logger().warn("No traversable costmap cell near robot.")
                return reachable

            start_x, start_y = nearest

        queue = deque()
        queue.append((start_x, start_y))
        reachable[start_y, start_x] = True

        while queue:
            x, y = queue.popleft()

            for nx, ny in (
                (x + 1, y),
                (x - 1, y),
                (x, y + 1),
                (x, y - 1),
            ):
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue

                if reachable[ny, nx]:
                    continue

                if not self.is_traversable_costmap_cell(nx, ny):
                    continue

                reachable[ny, nx] = True
                queue.append((nx, ny))

        reachable_count = int(np.count_nonzero(reachable))
        self.get_logger().info(f"Reachable global costmap cells: {reachable_count}")

        return reachable

    def find_nearest_traversable_costmap_cell(
        self,
        start_x,
        start_y,
        max_radius=10,
    ):
        for radius in range(1, max_radius + 1):
            for y in range(start_y - radius, start_y + radius + 1):
                for x in range(start_x - radius, start_x + radius + 1):
                    if self.is_traversable_costmap_cell(x, y):
                        return x, y

        return None

    # ------------------------------------------------------------
    # Planner validation
    # ------------------------------------------------------------
    def start_planner_validation(self, candidate_goals):
        self.pending_candidate_goals = list(candidate_goals)
        self.testing_plan = True

        self.try_next_candidate_goal()

    def try_next_candidate_goal(self):
        if not self.pending_candidate_goals:
            self.get_logger().warn("No plannable balanced frontier goals found.")
            self.testing_plan = False
            self.stop_robot()
            return

        x, y = self.pending_candidate_goals.pop(0)

        if self.is_blacklisted(x, y):
            self.try_next_candidate_goal()
            return

        self.current_test_goal_xy = (x, y)

        self.get_logger().info(
            f"Testing candidate with ComputePathToPose: x={x:.2f}, y={y:.2f}"
        )

        self.test_path_to_goal(x, y)

    def test_path_to_goal(self, x, y):
        if not self.plan_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error(
                f"Planner action server not available: {self.planner_action_name}"
            )
            self.testing_plan = False
            return

        goal_msg = ComputePathToPose.Goal()

        goal_msg.goal = PoseStamped()
        goal_msg.goal.header.frame_id = self.map_frame
        goal_msg.goal.header.stamp = self.get_clock().now().to_msg()

        goal_msg.goal.pose.position.x = float(x)
        goal_msg.goal.pose.position.y = float(y)
        goal_msg.goal.pose.position.z = 0.0

        robot_pose = self.get_robot_pose()

        if robot_pose is not None:
            robot_x, robot_y = robot_pose
            yaw = math.atan2(y - robot_y, x - robot_x)
        else:
            yaw = 0.0

        qz, qw = self.yaw_to_quaternion(yaw)

        goal_msg.goal.pose.orientation.x = 0.0
        goal_msg.goal.pose.orientation.y = 0.0
        goal_msg.goal.pose.orientation.z = qz
        goal_msg.goal.pose.orientation.w = qw

        # Let Nav2 use the current robot pose as start.
        try:
            goal_msg.use_start = False
        except Exception:
            pass

        send_future = self.plan_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self.plan_goal_response_callback)

    def plan_goal_response_callback(self, future):
        try:
            goal_handle = future.result()
        except Exception as e:
            self.get_logger().error(f"ComputePathToPose response failed: {e}")
            self.blacklist_current_test_goal()
            self.try_next_candidate_goal()
            return

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn("ComputePathToPose goal rejected.")
            self.blacklist_current_test_goal()
            self.try_next_candidate_goal()
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.plan_result_callback)

    def plan_result_callback(self, future):
        try:
            result = future.result()
        except Exception as e:
            self.get_logger().error(f"ComputePathToPose result failed: {e}")
            self.blacklist_current_test_goal()
            self.try_next_candidate_goal()
            return

        if result is None:
            self.get_logger().warn("ComputePathToPose returned no result.")
            self.blacklist_current_test_goal()
            self.try_next_candidate_goal()
            return

        x, y = self.current_test_goal_xy

        status = result.status

        if status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().warn(
                f"Candidate not plannable. Status={status}. "
                f"Blacklisting x={x:.2f}, y={y:.2f}"
            )
            self.blacklist_current_test_goal()
            self.try_next_candidate_goal()
            return

        path = result.result.path

        if len(path.poses) == 0:
            self.get_logger().warn(
                f"Planner returned empty path. Blacklisting x={x:.2f}, y={y:.2f}"
            )
            self.blacklist_current_test_goal()
            self.try_next_candidate_goal()
            return

        path_length = self.compute_path_length(path)

        if path_length > self.max_planned_path_length_m:
            self.get_logger().warn(
                f"Planned path too long for balanced exploration: "
                f"{path_length:.2f} m > {self.max_planned_path_length_m:.2f} m. "
                f"Blacklisting x={x:.2f}, y={y:.2f}"
            )
            self.blacklist_current_test_goal()
            self.try_next_candidate_goal()
            return

        self.get_logger().info(
            f"Candidate accepted by planner. "
            f"x={x:.2f}, y={y:.2f}, path_length={path_length:.2f} m"
        )

        self.testing_plan = False
        self.send_nav_goal(x, y)

    def compute_path_length(self, path):
        total = 0.0
        poses = path.poses

        for i in range(1, len(poses)):
            x0 = poses[i - 1].pose.position.x
            y0 = poses[i - 1].pose.position.y

            x1 = poses[i].pose.position.x
            y1 = poses[i].pose.position.y

            total += math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

        return total

    # ------------------------------------------------------------
    # Nav2 goal sending
    # ------------------------------------------------------------
    def send_nav_goal(self, x, y):
        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error(
                f"NavigateToPose action server not available: {self.nav_action_name}"
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
            f"Sending balanced Nav2 goal: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}"
        )

        self.active_goal = True
        self.goal_start_time = time.time()
        self.current_goal_xy = (x, y)

        send_future = self.nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self.nav_feedback_callback,
        )

        send_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        try:
            goal_handle = future.result()
        except Exception as e:
            self.get_logger().error(f"Goal response failed: {e}")
            self.active_goal = False
            self.goal_handle = None
            self.goal_start_time = None
            return

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn("NavigateToPose goal rejected.")
            self.blacklist_current_goal()
            self.active_goal = False
            self.goal_handle = None
            self.goal_start_time = None
            return

        self.get_logger().info("NavigateToPose goal accepted.")

        self.goal_handle = goal_handle

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)

    def nav_feedback_callback(self, feedback_msg):
        try:
            distance = feedback_msg.feedback.distance_remaining
            self.get_logger().debug(f"Distance remaining: {distance:.2f} m")
        except Exception:
            pass

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
                f"Navigation goal failed or was canceled. Status={status}"
            )

            self.blacklist_current_goal()

            self.active_goal = False
            self.goal_handle = None
            self.goal_start_time = None

    # ------------------------------------------------------------
    # Reachability in raw SLAM map
    # ------------------------------------------------------------
    def compute_reachable_free_mask(self, robot_cell):
        height, width = self.map_array.shape

        reachable = np.zeros((height, width), dtype=bool)

        start_x, start_y = robot_cell

        if not self.is_cell_in_bounds(start_x, start_y):
            return reachable

        if self.map_array[start_y, start_x] != 0:
            nearest = self.find_nearest_free_cell(start_x, start_y)

            if nearest is None:
                self.get_logger().warn("Could not find nearby free SLAM-map cell.")
                return reachable

            start_x, start_y = nearest

        queue = deque()
        queue.append((start_x, start_y))
        reachable[start_y, start_x] = True

        while queue:
            x, y = queue.popleft()

            for nx, ny in (
                (x + 1, y),
                (x - 1, y),
                (x, y + 1),
                (x, y - 1),
            ):
                if not self.is_cell_in_bounds(nx, ny):
                    continue

                if reachable[ny, nx]:
                    continue

                if self.map_array[ny, nx] != 0:
                    continue

                reachable[ny, nx] = True
                queue.append((nx, ny))

        return reachable

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
    # Coordinate helpers
    # ------------------------------------------------------------
    def is_cell_in_bounds(self, mx, my):
        height, width = self.map_array.shape

        return 0 <= mx < width and 0 <= my < height

    def compute_cluster_centroid(self, cluster):
        xs = [cell[0] for cell in cluster]
        ys = [cell[1] for cell in cluster]

        return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))

    def map_to_world(self, mx, my):
        if self.latest_map is None:
            return None

        resolution = self.latest_map.info.resolution
        origin = self.latest_map.info.origin.position

        wx = origin.x + (mx + 0.5) * resolution
        wy = origin.y + (my + 0.5) * resolution

        return wx, wy

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

    def world_to_global_costmap(self, wx, wy):
        if self.latest_global_costmap is None or self.global_costmap_array is None:
            return None

        resolution = self.latest_global_costmap.info.resolution
        origin = self.latest_global_costmap.info.origin.position

        mx = int((wx - origin.x) / resolution)
        my = int((wy - origin.y) / resolution)

        height, width = self.global_costmap_array.shape

        if mx < 0 or my < 0 or mx >= width or my >= height:
            return None

        return mx, my

    def get_robot_pose(self):
        frames_to_try = []

        for frame in [self.robot_base_frame, "base_link", "base_footprint"]:
            if frame not in frames_to_try:
                frames_to_try.append(frame)

        for frame in frames_to_try:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    frame,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.5),
                )

                x = transform.transform.translation.x
                y = transform.transform.translation.y

                return x, y

            except Exception:
                pass

        self.get_logger().warn(
            f"TF lookup failed from {self.map_frame} to {frames_to_try}"
        )

        return None

    def yaw_to_quaternion(self, yaw):
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)

        return qz, qw

    # ------------------------------------------------------------
    # Blacklist / cancel / stop
    # ------------------------------------------------------------
    def blacklist_current_goal(self):
        if self.current_goal_xy is None:
            return

        self.get_logger().warn(
            f"Blacklisting navigation goal: "
            f"x={self.current_goal_xy[0]:.2f}, "
            f"y={self.current_goal_xy[1]:.2f}"
        )

        self.blacklisted_goals.append(self.current_goal_xy)

    def blacklist_current_test_goal(self):
        if self.current_test_goal_xy is None:
            return

        self.get_logger().warn(
            f"Blacklisting test goal: "
            f"x={self.current_test_goal_xy[0]:.2f}, "
            f"y={self.current_test_goal_xy[1]:.2f}"
        )

        self.blacklisted_goals.append(self.current_test_goal_xy)

    def is_blacklisted(self, x, y):
        for bx, by in self.blacklisted_goals:
            distance = math.sqrt((x - bx) ** 2 + (y - by) ** 2)

            if distance < self.blacklist_radius_m:
                return True

        return False

    def cancel_current_goal(self):
        if self.goal_handle is not None:
            self.get_logger().info("Canceling current Nav2 goal.")
            self.goal_handle.cancel_goal_async()

        self.active_goal = False
        self.goal_handle = None
        self.goal_start_time = None

    def goal_timed_out(self):
        if self.goal_start_time is None:
            return False

        elapsed = time.time() - self.goal_start_time

        return elapsed > self.goal_timeout_sec

    def stop_robot(self):
        twist = Twist()
        self.cmd_vel_pub.publish(twist)

    # ------------------------------------------------------------
    # Optional scan
    # ------------------------------------------------------------
    def rotate_360(self):
        self.get_logger().info("Performing optional 360-degree scan.")

        if abs(self.scan_angular_speed) < 1e-6:
            return

        twist = Twist()
        twist.angular.z = self.scan_angular_speed

        duration = 2.0 * math.pi / abs(self.scan_angular_speed)
        start = time.time()

        while rclpy.ok() and time.time() - start < duration:
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.05)

        self.stop_robot()


def main(args=None):
    rclpy.init(args=args)

    node = ExploreNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info("Explorer interrupted by user.")

    finally:
        node.cancel_current_goal()
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()