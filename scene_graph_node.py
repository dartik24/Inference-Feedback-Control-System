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
        self.object_match_distance_m = 0.75
        self.near_threshold_m = 1.0

        self.memory_file = "world_scene_graph.json"

        self.world_memory = {
            "objects": {},
            "edges": [],
            "rooms": {}
        }

        self.next_world_object_id = 0

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

        self.load_world_memory()

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
        6. Updates persistent object memory.
        7. Infers global object edges.
        8. Infers room clusters.
        9. Saves and publishes the graph.
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

        observed_graph = self.build_observed_graph(
            self.latest_rgb,
            self.latest_depth,
            self.camera_info
        )

        self.update_world_memory(observed_graph)

        self.world_memory["edges"] = self.infer_world_edges()
        self.world_memory["rooms"] = self.infer_room_clusters()

        self.save_world_memory()
        self.publish_scene_graph()

        self.get_logger().info(
            f"Graph updated. Objects: {len(self.world_memory['objects'])}, "
            f"Rooms: {len(self.world_memory['rooms'])}"
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

        return {
            "nodes": objects
        }

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

    def update_world_memory(self, observed_graph):
        """Merge current detected objects into persistent memory."""

        now = time.time()

        for obs in observed_graph["nodes"]:
            if obs["position_map_frame"] is None:
                continue

            match_id = self.find_matching_object(obs)

            if match_id is None:
                self.create_world_object(obs, now)
            else:
                self.update_world_object(match_id, obs, now)

    def find_matching_object(self, obs):
        """Match an observation to an existing object by label and distance."""

        obs_label = obs["label"]
        obs_pos = obs["position_map_frame"]

        best_id = None
        best_dist = float("inf")

        for obj_id, obj in self.world_memory["objects"].items():
            if obj["label"] != obs_label:
                continue

            obj_pos = obj["position_map_frame"]

            dx = obs_pos["x"] - obj_pos["x"]
            dy = obs_pos["y"] - obj_pos["y"]
            dz = obs_pos.get("z", 0.0) - obj_pos.get("z", 0.0)

            dist = math.sqrt(dx * dx + dy * dy + dz * dz)

            if dist < self.object_match_distance_m and dist < best_dist:
                best_dist = dist
                best_id = obj_id

        return best_id

    def create_world_object(self, obs, now):
        """Create a new persistent object."""

        obj_id = f"world_obj_{self.next_world_object_id}"
        self.next_world_object_id += 1

        self.world_memory["objects"][obj_id] = {
            "id": obj_id,
            "label": obs["label"],
            "confidence": obs["confidence"],
            "position_map_frame": obs["position_map_frame"],
            "first_seen": now,
            "last_seen": now,
            "times_seen": 1,
            "room_id": None,
            "observations": [
                {
                    "timestamp": now,
                    "bbox": obs["bbox"],
                    "confidence": obs["confidence"],
                    "robot_location": self.latest_pose
                }
            ]
        }

    def update_world_object(self, obj_id, obs, now):
        """Update an existing persistent object."""

        obj = self.world_memory["objects"][obj_id]

        old_pos = obj["position_map_frame"]
        new_pos = obs["position_map_frame"]

        alpha = 0.25

        obj["position_map_frame"] = {
            "x": (1 - alpha) * old_pos["x"] + alpha * new_pos["x"],
            "y": (1 - alpha) * old_pos["y"] + alpha * new_pos["y"],
            "z": (1 - alpha) * old_pos.get("z", 0.0) + alpha * new_pos.get("z", 0.0),
            "frame_id": self.map_frame
        }

        obj["confidence"] = max(obj["confidence"], obs["confidence"])
        obj["last_seen"] = now
        obj["times_seen"] += 1

        obj["observations"].append({
            "timestamp": now,
            "bbox": obs["bbox"],
            "confidence": obs["confidence"],
            "robot_location": self.latest_pose
        })

        obj["observations"] = obj["observations"][-10:]

    def infer_world_edges(self):
        """Infer global object-object spatial relationships."""

        edges = []
        objects = list(self.world_memory["objects"].values())

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
                    edges.append({
                        "source": a["id"],
                        "relation": "near",
                        "target": b["id"],
                        "distance_m": float(dist)
                    })
                    edges.append({
                        "source": b["id"],
                        "relation": "near",
                        "target": a["id"],
                        "distance_m": float(dist)
                    })

                if ax < bx:
                    edges.append({"source": a["id"], "relation": "west_of", "target": b["id"]})
                    edges.append({"source": b["id"], "relation": "east_of", "target": a["id"]})
                else:
                    edges.append({"source": a["id"], "relation": "east_of", "target": b["id"]})
                    edges.append({"source": b["id"], "relation": "west_of", "target": a["id"]})

                if ay < by:
                    edges.append({"source": a["id"], "relation": "south_of", "target": b["id"]})
                    edges.append({"source": b["id"], "relation": "north_of", "target": a["id"]})
                else:
                    edges.append({"source": a["id"], "relation": "north_of", "target": b["id"]})
                    edges.append({"source": b["id"], "relation": "south_of", "target": a["id"]})

        return edges

    def infer_room_clusters(self):
        """Infer room-like clusters from tightly connected near edges."""

        objects = self.world_memory["objects"]
        edges = self.world_memory["edges"]

        adjacency = {obj_id: [] for obj_id in objects.keys()}

        for edge in edges:
            if edge["relation"] != "near":
                continue

            source = edge["source"]
            target = edge["target"]

            if source in adjacency and target in adjacency:
                adjacency[source].append(target)
                adjacency[target].append(source)

        components = self.find_connected_components(adjacency)

        rooms = {}

        for i, component in enumerate(components):
            room_id = f"room_{i}"

            centroid = self.compute_room_centroid(component)
            room_edges = self.get_edges_for_room(component)

            rooms[room_id] = {
                "id": room_id,
                "room_label": "unknown_room",
                "object_ids": component,
                "object_count": len(component),
                "centroid": centroid,
                "edges": room_edges
            }

            for obj_id in component:
                self.world_memory["objects"][obj_id]["room_id"] = room_id

        return rooms

    def find_connected_components(self, adjacency):
        """Find connected components in an adjacency list."""

        visited = set()
        components = []

        for start_id in adjacency.keys():
            if start_id in visited:
                continue

            stack = [start_id]
            component = []

            while stack:
                current = stack.pop()

                if current in visited:
                    continue

                visited.add(current)
                component.append(current)

                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        stack.append(neighbor)

            components.append(component)

        return components

    def compute_room_centroid(self, object_ids):
        """Compute average x/y map position for a room cluster."""

        xs = []
        ys = []

        for obj_id in object_ids:
            obj = self.world_memory["objects"][obj_id]
            pos = obj["position_map_frame"]

            xs.append(pos["x"])
            ys.append(pos["y"])

        if len(xs) == 0:
            return {
                "x": 0.0,
                "y": 0.0,
                "frame_id": self.map_frame
            }

        return {
            "x": sum(xs) / len(xs),
            "y": sum(ys) / len(ys),
            "frame_id": self.map_frame
        }

    def get_edges_for_room(self, object_ids):
        """Return all global edges whose source and target are inside a room."""

        object_set = set(object_ids)
        room_edges = []

        for edge in self.world_memory["edges"]:
            if edge["source"] in object_set and edge["target"] in object_set:
                room_edges.append(edge)

        return room_edges

    def publish_scene_graph(self):
        """Publish full graph to ROS topic: scene_graph."""

        msg = String()
        msg.data = json.dumps(self.world_memory, indent=2)
        self.scene_graph_pub.publish(msg)

    def save_world_memory(self):
        """Save persistent graph to disk."""

        with open(self.memory_file, "w") as f:
            json.dump(self.world_memory, f, indent=2)

    def load_world_memory(self):
        """Load persistent graph from disk if available."""

        try:
            with open(self.memory_file, "r") as f:
                self.world_memory = json.load(f)

            if "objects" not in self.world_memory:
                self.world_memory["objects"] = {}
            if "edges" not in self.world_memory:
                self.world_memory["edges"] = []
            if "rooms" not in self.world_memory:
                self.world_memory["rooms"] = {}

            max_id = -1

            for obj_id in self.world_memory["objects"].keys():
                try:
                    number = int(obj_id.replace("world_obj_", ""))
                    max_id = max(max_id, number)
                except ValueError:
                    pass

            self.next_world_object_id = max_id + 1

            self.get_logger().info(
                f"Loaded memory: "
                f"{len(self.world_memory['objects'])} objects, "
                f"{len(self.world_memory['rooms'])} rooms."
            )

        except FileNotFoundError:
            self.get_logger().info("No existing world memory file found.")
        except Exception as e:
            self.get_logger().warn(f"Could not load world memory: {e}")


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