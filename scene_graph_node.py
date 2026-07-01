#!/usr/bin/env python3

import json
import time
import math

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String

from cv_bridge import CvBridge
import numpy as np
from ultralytics import YOLO

import tf2_ros
from tf2_geometry_msgs import do_transform_point


class WorldSceneGraphNode(Node):
    def __init__(self):
        super().__init__("world_scene_graph_node")

        self.bridge = CvBridge()

        self.latest_rgb = None
        self.latest_depth = None
        self.camera_info = None
        self.latest_pose = None

        self.rgb_topic = "/camera/color/image_raw"
        self.depth_topic = "/camera/depth/image_rect_raw"
        self.camera_info_topic = "/camera/color/camera_info"

        self.camera_frame = "camera_color_optical_frame"
        self.map_frame = "map"
        self.base_frame = "base_link"

        self.update_period_sec = 3.0
        self.conf_threshold = 0.45
        self.near_threshold_m = 1.0

        self.memory_file = "world_scene_graph.json"

        self.scene_graph = {
            "timestamp": None,
            "robot_location": None,
            "objects": [],
            "relations": []
        }

        self.model = YOLO("yolov8n.pt")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(Image, self.rgb_topic, self.rgb_callback, 10)
        self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)

        self.scene_graph_pub = self.create_publisher(
            String,
            "scene_graph",
            10
        )

        self.timer = self.create_timer(
            self.update_period_sec,
            self.update_scene_graph
        )

        self.load_scene_graph()

        self.get_logger().info("World scene graph node started.")

    def rgb_callback(self, msg):
        """Store latest RGB camera image."""
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )
        except Exception as e:
            self.get_logger().error(f"RGB conversion failed: {e}")

    def depth_callback(self, msg):
        """Store latest depth image."""
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="passthrough"
            )
        except Exception as e:
            self.get_logger().error(f"Depth conversion failed: {e}")

    def camera_info_callback(self, msg):
        """Store latest camera intrinsics."""
        self.camera_info = msg

    def update_robot_pose_from_tf(self):
        """
        Gets the robot pose from TF instead of /amcl_pose.

        This looks up:

            map -> base_link

        If your robot only has odom -> base_link, change:

            self.map_frame = "odom"
        """

        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time()
            )

            t = transform.transform.translation
            q = transform.transform.rotation

            self.latest_pose = {
                "x": float(t.x),
                "y": float(t.y),
                "z": float(t.z),
                "qx": float(q.x),
                "qy": float(q.y),
                "qz": float(q.z),
                "qw": float(q.w),
                "frame_id": self.map_frame
            }

            return True

        except Exception as e:
            self.get_logger().warn(f"Could not get robot pose from TF: {e}")
            return False

    def update_scene_graph(self):
        """
        Main update loop.

        Each cycle:
        1. Checks camera data.
        2. Gets robot pose from TF.
        3. Detects objects.
        4. Estimates 3D object positions.
        5. Transforms objects into map frame.
        6. Logs only the currently observed objects and their relations.
        7. Saves and publishes the graph.
        """

        if self.latest_rgb is None:
            self.get_logger().warn("No RGB image yet.")
            return

        if self.latest_depth is None:
            self.get_logger().warn("No depth image yet.")
            return

        if self.camera_info is None:
            self.get_logger().warn("No camera info yet.")
            return

        if not self.update_robot_pose_from_tf():
            self.get_logger().warn("No robot pose from TF yet.")
            return

        objects = self.build_observed_graph(
            self.latest_rgb,
            self.latest_depth,
            self.camera_info
        )

        relations = self.infer_observed_relations(objects)

        self.scene_graph = {
            "timestamp": time.time(),
            "robot_location": self.latest_pose,
            "objects": objects,
            "relations": relations
        }

        self.save_scene_graph()
        self.publish_scene_graph()
        self.log_observed_scene()

        self.get_logger().info(
            f"Scene graph updated. Objects: {len(objects)}, "
            f"Relations: {len(relations)}"
        )

    def build_observed_graph(self, rgb_image, depth_image, camera_info):
        """Build a temporary observation graph from the current camera frame."""

        objects = self.detect_objects(rgb_image)

        objects = self.estimate_camera_positions(
            objects,
            depth_image,
            camera_info
        )

        objects = self.transform_objects_to_map(objects)

        return [
            obj
            for obj in objects
            if obj["position_map_frame"] is not None
        ]

    def detect_objects(self, rgb_image):
        """Run YOLO object detection."""

        results = self.model(rgb_image, verbose=False)[0]
        objects = []

        for i, box in enumerate(results.boxes):
            conf = float(box.conf[0])

            if conf < self.conf_threshold:
                continue

            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            center_u = int((x1 + x2) / 2)
            center_v = int((y1 + y2) / 2)

            objects.append({
                "id": f"obs_{i}",
                "label": label,
                "confidence": conf,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "center_pixel": {
                    "u": center_u,
                    "v": center_v
                },
                "position_camera_frame": None,
                "position_map_frame": None
            })

        return objects

    def estimate_camera_positions(self, objects, depth_image, camera_info):
        """Estimate 3D object positions in camera frame using depth."""

        fx = camera_info.k[0]
        fy = camera_info.k[4]
        cx = camera_info.k[2]
        cy = camera_info.k[5]

        height, width = depth_image.shape[:2]

        for obj in objects:
            u = obj["center_pixel"]["u"]
            v = obj["center_pixel"]["v"]

            if u < 0 or u >= width or v < 0 or v >= height:
                continue

            depth = self.get_stable_depth(depth_image, u, v)

            if depth is None:
                continue

            x = (u - cx) * depth / fx
            y = (v - cy) * depth / fy
            z = depth

            obj["position_camera_frame"] = {
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "frame_id": self.camera_frame
            }

        return objects

    def get_stable_depth(self, depth_image, u, v, window_size=7):
        """Use median depth around the center pixel to reduce noise."""

        h, w = depth_image.shape[:2]
        half = window_size // 2

        u1 = max(0, u - half)
        u2 = min(w, u + half + 1)
        v1 = max(0, v - half)
        v2 = min(h, v + half + 1)

        patch = depth_image[v1:v2, u1:u2].astype(np.float32)

        patch = patch[np.isfinite(patch)]
        patch = patch[patch > 0]

        if len(patch) == 0:
            return None

        depth = float(np.median(patch))

        if depth > 20.0:
            depth = depth / 1000.0

        return depth

    def transform_objects_to_map(self, objects):
        """Transform detected objects from camera frame into map frame."""

        for obj in objects:
            pos = obj["position_camera_frame"]

            if pos is None:
                continue

            point_camera = PointStamped()
            point_camera.header.frame_id = self.camera_frame
            point_camera.header.stamp = rclpy.time.Time().to_msg()
            point_camera.point.x = pos["x"]
            point_camera.point.y = pos["y"]
            point_camera.point.z = pos["z"]

            try:
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    self.camera_frame,
                    rclpy.time.Time()
                )

                point_map = do_transform_point(point_camera, transform)

                obj["position_map_frame"] = {
                    "x": float(point_map.point.x),
                    "y": float(point_map.point.y),
                    "z": float(point_map.point.z),
                    "frame_id": self.map_frame
                }

            except Exception as e:
                self.get_logger().warn(
                    f"Could not transform object to map frame: {e}"
                )

        return objects

    def infer_observed_relations(self, objects):
        """Infer spatial relationships between objects observed in this update."""

        relations = []

        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                a = objects[i]
                b = objects[j]

                ax = a["position_map_frame"]["x"]
                ay = a["position_map_frame"]["y"]
                bx = b["position_map_frame"]["x"]
                by = b["position_map_frame"]["y"]

                dx = ax - bx
                dy = ay - by
                dist = math.sqrt(dx * dx + dy * dy)

                if dist < self.near_threshold_m:
                    relations.append({
                        "source": a["id"],
                        "source_label": a["label"],
                        "relation": "near",
                        "target": b["id"],
                        "target_label": b["label"],
                        "distance_m": float(dist)
                    })
                    relations.append({
                        "source": b["id"],
                        "source_label": b["label"],
                        "relation": "near",
                        "target": a["id"],
                        "target_label": a["label"],
                        "distance_m": float(dist)
                    })

                x_relation = "west_of" if ax < bx else "east_of"
                inverse_x_relation = "east_of" if ax < bx else "west_of"
                y_relation = "south_of" if ay < by else "north_of"
                inverse_y_relation = "north_of" if ay < by else "south_of"

                relations.append({
                    "source": a["id"],
                    "source_label": a["label"],
                    "relation": x_relation,
                    "target": b["id"],
                    "target_label": b["label"],
                    "distance_m": float(dist)
                })
                relations.append({
                    "source": b["id"],
                    "source_label": b["label"],
                    "relation": inverse_x_relation,
                    "target": a["id"],
                    "target_label": a["label"],
                    "distance_m": float(dist)
                })
                relations.append({
                    "source": a["id"],
                    "source_label": a["label"],
                    "relation": y_relation,
                    "target": b["id"],
                    "target_label": b["label"],
                    "distance_m": float(dist)
                })
                relations.append({
                    "source": b["id"],
                    "source_label": b["label"],
                    "relation": inverse_y_relation,
                    "target": a["id"],
                    "target_label": a["label"],
                    "distance_m": float(dist)
                })

        return relations

    def log_observed_scene(self):
        """Log a compact human-readable version of the current observation."""

        for obj in self.scene_graph["objects"]:
            pos = obj["position_map_frame"]
            self.get_logger().info(
                f"Observed {obj['label']} ({obj['id']}) at "
                f"x={pos['x']:.2f}, y={pos['y']:.2f}, z={pos['z']:.2f} "
                f"in {pos['frame_id']}"
            )

        for rel in self.scene_graph["relations"]:
            self.get_logger().info(
                f"Relation: {rel['source_label']} ({rel['source']}) "
                f"{rel['relation']} {rel['target_label']} "
                f"({rel['target']}) at {rel['distance_m']:.2f} m"
            )

    def publish_scene_graph(self):
        """Publish full graph to ROS topic: scene_graph."""

        msg = String()
        msg.data = json.dumps(self.scene_graph, indent=2)
        self.scene_graph_pub.publish(msg)

    def save_scene_graph(self):
        """Save the latest observed scene graph to disk."""

        with open(self.memory_file, "w") as f:
            json.dump(self.scene_graph, f, indent=2)

    def load_scene_graph(self):
        """Load the most recent observed scene graph from disk if available."""

        try:
            with open(self.memory_file, "r") as f:
                loaded_graph = json.load(f)

            self.scene_graph = {
                "timestamp": loaded_graph.get("timestamp"),
                "robot_location": loaded_graph.get("robot_location"),
                "objects": self.normalize_loaded_objects(
                    loaded_graph.get("objects", [])
                ),
                "relations": loaded_graph.get(
                    "relations",
                    loaded_graph.get("edges", [])
                )
            }

            self.get_logger().info(
                f"Loaded latest scene graph: "
                f"{len(self.scene_graph['objects'])} objects, "
                f"{len(self.scene_graph['relations'])} relations."
            )

        except FileNotFoundError:
            self.get_logger().info("No existing scene graph file found.")
        except Exception as e:
            self.get_logger().warn(f"Could not load scene graph: {e}")

    def normalize_loaded_objects(self, objects):
        """Accept old dict-based files, but keep the active schema as a list."""

        if isinstance(objects, dict):
            return list(objects.values())

        if isinstance(objects, list):
            return objects

        return []


def main(args=None):
    rclpy.init(args=args)

    node = WorldSceneGraphNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
