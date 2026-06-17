#ROS2 CORE
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from contextlib import contextmanager

#NAVIGATION AND MAPPING
from geometry_msgs.msg import Twist
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseWithCovarianceStamped

#SPEECH AND AUDIO
import speech_recognition as sr
import pyaudio
import pyttsx3
from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play
import simpleaudio as sa
from openai import OpenAI
import pygame

#PYTHON CORE
import time
import os
import sys
import numpy as np
import json
import math
from math import sqrt
import threading
from threading import Lock
from std_msgs.msg import String
from queue import Queue
from datetime import datetime
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

#VISION
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

#COMPANION APP
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

@contextmanager
def ignore_stderr():
    devnull = None
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        stderr = os.dup(2)
        sys.stderr.flush()
        os.dup2(devnull, 2)
        try:
            yield
        finally:
            os.dup2(stderr, 2)
            os.close(stderr)
    finally:
        if devnull is not None:
            os.close(devnull)

def get_respeaker_device_id():
    with ignore_stderr():
        p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')
    device_id = -1
    for i in range(num_devices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            if "ReSpeaker" in p.get_device_info_by_host_api_device_index(0, i).get('name'):
                device_id = i

    return device_id

ACTIONS = []
actions_lock = Lock()
ROS_NODE = None

class VoiceControlNode(Node):
    def __init__(self):
        super().__init__('voice_control_node')
        self.publisher_ = self.create_publisher(Twist, '/stretch/cmd_vel', 10)
        self.image_subscriber = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10
        )
        self.map_pub = self.create_publisher(String, '/map_server/map', 10)

        self.pose_subscriber = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.update_current_pose,
            10
        )
        self.current_pose = None  
        self.goal_threshold = 0.2  

        # Drive engine
        self.engine = pyttsx3.init()
        self.engine_lock = Lock()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone(get_respeaker_device_id())
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.twist = Twist()
        self.twist.angular.z = 0.0
        self.twist.linear.x = 0.0
        self.bridge = CvBridge()
        self.get_logger().info("Voice control node has been started.")

        # Command engine
        self.task_queue = Queue()
        self.task_queue_lock = Lock()
        self.current_task = None
        self.is_navigating = False
        self.current_goal_handle = None
        self.taskid = 0
        self.last_position = (0.0, 0.0)
        self.destination = None
        self.splitter = None
        self.comm = None
        self.triggers = ["stretch", "robot", "buddy", "companion", "pal", "mate", "amigo", "friend"]

        # Configuration and prompt loading
        with open(Path(get_package_share_directory('turtlebot4_voice_control')) / 'Config' / 'config.json', "r", encoding="utf-8") as f:
            self.config = json.load(f)

        with open(Path(get_package_share_directory('turtlebot4_voice_control')) / self.config['paths']['infer_prompt'], "r", encoding="utf-8") as f:
            self.prompt = (
                "You are a voice request interpreter.\n\n"
                "Use the following configuration:\n\n"
                + json.dumps(json.load(f), indent=2)
            )

        with open(Path(get_package_share_directory('turtlebot4_voice_control')) / self.config['paths']['confirm_prompt'], "r", encoding="utf-8") as f:
            self.conPrompt = (
                "You are an input classifier.\n\n"
                "Use the following configuration:\n\n"
                + json.dumps(json.load(f), indent=2)
                + "\n\nIMPORTANT: Your response must consist of exactly one label and nothing else."
            )
       
        # Preset Locations
        for room, coords in self.config["locations"].items():
            setattr(self, room, coords)

        # Sound engine
        pygame.mixer.init()
        # self.music_file = '/home/hello-robot/Downloads/01 - The Pink Panther Theme.mp3'  
        # self.waiting_music = '/home/hello-robot/Downloads/Waiting.mp3'
        self.music_volume = 0.5  
        self.speaking = False
        self.speak_lock = Lock()
        self.state_lock = Lock()

        # Companion App
        self.actions = []
        self.actionnum = 0
        self.addactcount = 0
        self.gid = 0

        try:
            pygame.mixer.music.set_volume(0)  
        except Exception as e:
            self.get_logger().error(f"Error initializing music playback: {e}")

        self.person_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
        self.current_frame = None
        self.person_detected = False

        threading.Thread(target=self.listen_for_commands, daemon=True).start()
        threading.Thread(target=self.process_task_queue, daemon=True).start()
        threading.Thread(target=self.publish_cmd_vel, daemon=True).start()

        self.load_map()

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        if not self.nav_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('NavigateToPose action server not available.')

    # Adds an action to the action topic, will always be set to be proposed untill passed into add_to_tasl_queue.
    def add_action(self, task):
        self.actions = self.actions + [
            {
                "id" : self.actionnum + 1,
                "pos" : self.actionnum + 1,
                "tnum" : task[2],
                "name" : str(task),
                "status" : "ongoing" if(self.actionnum + 1) == 1 else "pending",
                "destination" : str(task[1]),
                "phase" : "proposed",
                "started_at" : None,
            }
        ]
        print(self.actions)
        self.sync_actions_for_http()

    # Updates action list to remove the first one and update the remaining ones accordingly.
    def update_action(self):
        with actions_lock:
            if not self.actions:
                self.sync_actions_for_http()
                return

            self.actions.pop(0)

            for idx, action in enumerate(self.actions):
                action["pos"] = idx + 1
                action["status"] = "ongoing" if idx == 0 else "pending"
                action["started_at"] = None

        self.sync_actions_for_http()

    # Fetches ids for every action from first to last
    def _queued_action_ids_in_order(self):
        with self.task_queue_lock:
            return [t[2] for t in list(self.task_queue.queue)]

    # Syncs action list with the companion app, use this function to update UI if changes are made to robot side action list.
    def sync_actions_for_http(self):
        global ACTIONS 
        queue_ids = self._queued_action_ids_in_order()
        pos_map = {aid: i for i, aid in enumerate(queue_ids)}

        with actions_lock:
            out = []
            for a in self.actions:
                aa = dict(a)
                aa["pos"] = (pos_map[a["id"]] + 1) if a["id"] in pos_map else None
                aa["status"] = "ongoing" if self.actions.index(a) == 0 else "pending"
                out.append(aa)

            ACTIONS = out

    # Function is ran when a new list of actions is recieved from companion app, will reorder the task queue and action list based on a new order of ids. IDS FOR ACTION LIST AND TASK QUEUE BOTH ON THE ROBOT AND COMPANION APP MUST ALWAYS BE SYNCED UP FOR THIS TO WORK.
    def reorder_tasks(self, OrderedIds):
        with self.task_queue_lock:
            print(list(self.task_queue.queue))
            pending = []
            while True:
                try:
                    pending.append(self.task_queue.get_nowait())
                except Exception:
                    break

            ids = {t[2]: t for t in pending}
            npending = []

            for aid in OrderedIds:
                t = ids.pop(aid, None)
                if t is not None:
                    npending.append(t)

            for t in pending:
                if t[2] in ids:
                    npending.append(ids.pop(t[2]))

            for task in npending:
                self.task_queue.put(task)
        
        action_ids = {a["id"]: a for a in self.actions}
        na = []

        for aid in  OrderedIds:
            if aid in action_ids:
                na.append(action_ids.pop(aid))
        
        na.extend(action_ids.values())
        self.actions = na
        self.sync_actions_for_http()
        print(list(self.task_queue.queue))
        self.get_logger().info("Tasks reordering complete.")

    # Load map from default file
    def load_map(self):
        map_path = os.getenv('HELLO_FLEET_PATH', '') + '/maps/Study2_Map.yaml'
        if not os.path.exists(map_path):
            self.get_logger().error(f"Map file not found at: {map_path}")
            return

        self.get_logger().info(f"Loading map from: {map_path}")
        map_msg = String()
        map_msg.data = map_path
        self.map_pub.publish(map_msg)
        time.sleep(1)

    # Person follow image callback
    def image_callback(self, msg):
        """Callback to process camera feed."""
        try:
            self.current_frame = cv2.rotate(self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8'),cv2.ROTATE_90_CLOCKWISE)
        except Exception as e:
            self.get_logger().error(f"Failed to convert image message to OpenCV format: {e}")

    # Base navigation function
    def publish_cmd_vel(self):
        while rclpy.ok():
            self.publisher_.publish(self.twist)
            time.sleep(0.1)

    def api_initializer(self):
        completion = self.client.chat.completions.create(
            model=self.config["openai"]["model"],
            messages=[
                {"role": "system",
                 "content": self.prompt},
                {
                    "role": "user",
                    "content": ""
                }
            ],
            temperature=0,
            max_tokens=1,
        )

    # Voice input and processing
    def listen_for_commands(self):
        with self.microphone as source:
            self.api_initializer()
            self.get_logger().info("Calibrating microphone (0.8s)... ")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.8)
            self.get_logger().info("Listener Thread Listening... ")
            self.speak("I am listening")

            ce = 0

            while True:
                if self.speaking:
                    self.get_logger().info("Currently speaking.")
                    time.sleep(0.5)
                    continue
                
                try:
                    audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=6.0)
                except sr.WaitTimeoutError:
                    continue
                
                try:
                    command = self.recognizer.recognize_google(audio).lower().strip()
                    isRequest = any(trigger in command for trigger in self.triggers)
                    
                    if self.splitter is None:
                        if command and isRequest:
                            self.get_logger().info(f"Recognized command: {command}")
                            self.speak("Got it")
                            self.handle_command(command)
                        else:
                            continue
                    else:
                        if command:
                            self.get_logger().info(f"Recognized command: {command}")
                            self.speak("Got it")
                            self.handle_command(command)
                        else:
                            continue

                except sr.UnknownValueError:
                    self.get_logger().info("None Recieved")

                except sr.RequestError as e:
                    self.get_logger().error(f"Speech recognition error: {e}")
                    ce += 1
                    time.sleep(0.5)

                time.sleep(0.5)

    # Command processing and handling
    def handle_command(self, command):
        subcommands = self.command_interpretation(command)
        if subcommands:
            for subcommand in subcommands:
                targets = [x.strip() for x in subcommand.split(",")]
                task_type, task_dest, task_id = targets

                if len(targets) > 3:
                    self.speak("I dint fully understand, could you say that again?") 

                self.add_task_to_queue((task_type, task_dest))

    # Voice command interpretation
    # Edit 1: after initial interpretation now sets a flag through the splitter variable and if ran with flag set will enter confirmation mode. Confirmation mode will instead request a response and distinguish between it being a confirmation clarification or rejection. 
    # Edit 2: modified the strucuture to now update and sync action list immediatly after intial proposed commands are generated. If modified or rejected proposed action list is regenerated or deleted accordingly. 
    def command_interpretation(self, command):
        if self.splitter == None:
            # self.speak("Give me a moment to think")
            self.comm = command
            completion = self.client.chat.completions.create(
                model=self.config["openai"]["model"],
                messages=[
                    {"role": "system",
                    "content": self.prompt},
                    {
                        "role": "user",
                        "content": command
                    }
                ],
                temperature=0,
                max_tokens=100,
            )
            print(completion.choices[0].message.content)
            try:
                self.splitter = completion.choices[0].message.content.splitlines()
                if self.splitter[1]:
                    self.speak(self.splitter[1])
                else:
                    self.speak("Please confirm that that is what you want me to do.")
                if self.splitter[0]:
                    tg = 0
                    self.splitter[0] = self.splitter[0].split(" then ")
                    while self.addactcount != 0:
                        self.actions.pop()
                        self.addactcount = self.addactcount - 1
                        self.actionnum = self.actionnum - 1
                    for subcommand in self.splitter[0]:
                        targets = subcommand.split(",")
                        if len(targets) > 3:
                            self.speak("I dont think I got that, could you say that again?") 
                        self.add_action((targets[0], targets[1], self.gid + int(targets[2])))
                        self.actionnum = self.actionnum + 1
                        self.addactcount = self.addactcount + 1
                        tg = int(targets[2])
                    self.gid = self.gid + tg
                    print(self.gid)
                return None
            except:
                return self.command_interpretation(self.comm)

        else:
            completion = self.client.chat.completions.create(
                model=self.config["openai"]["model"],
                messages=[
                    {"role": "system",
                    "content": self.conPrompt},
                    {
                        "role": "user",
                        "content": command
                    }
                ],
                temperature=0,
                max_tokens=2,
            )
            print(completion.choices[0].message.content)
            if completion.choices[0].message.content == "Confirmation" or completion.choices[0].message.content == "confirmation":
                ret = self.splitter[0]
                self.splitter = None
                self.comm = None
                self.addactcount = 0
                print(list(ret))
                return ret
            elif completion.choices[0].message.content == "Clarification" or completion.choices[0].message.content == "clarification":
                full_command = self.comm + " " + command
                self.splitter = None
                return self.command_interpretation(full_command)
            else:
                while self.addactcount != 0:
                    self.actions.pop()
                    self.addactcount = self.addactcount - 1
                    self.actionnum = self.actionnum - 1
                self.sync_actions_for_http()
                self.splitter = None
                self.comm = None
                self.addactcount = 0
                self.speak("Task plan removed.")
                return None


    # Universal halt to all activity and queue clear
    def stop_all_tasks(self):
        # self.fade_out_and_stop_music()
        with self.task_queue_lock:
            while not self.task_queue.empty():
                try:
                    self.task_queue.get_nowait()
                except Exception:
                    break
        self.is_navigating = False
        self.current_task = None
        self.addactcount = 0
        self.actionnum = 0
        self.taskid = 0
        if self.current_goal_handle:
            self.current_goal_handle.cancel_goal_async()
            self.current_goal_handle = None
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.publisher_.publish(self.twist)
        self.speak("All tasks stopped.")

    # Addition of current task to the task queue
    def add_task_to_queue(self, task):
        with self.task_queue_lock:
            self.taskid = self.taskid + 1
            self.task_queue.put((task[0], task[1], self.taskid))
            print(list(self.task_queue.queue))

        with actions_lock:
            for a in self.actions:
                if a["id"] == self.taskid:
                    a["phase"] = "queued"
                    break
                    
        self.sync_actions_for_http()
        # self.speak("Task added to the queue.")

    # Command execution off the top of the queue
    def process_task_queue(self):
        while rclpy.ok():
            if not self.is_navigating and not self.task_queue.empty():
                task = self.task_queue.get()
                    
                task_type, task_dest, task_id = task
                print(task_type)
                self.current_task = task_type
                self.destination = task_dest
                if task_type == "go":
                    self.navigate_to(getattr(self, task_dest, "Room not found")[0], getattr(self, task_dest, "Room not found")[1])
                    self.speak("Going to the " + task_dest)
                elif task_type == "wait":
                    self.navigate_to(getattr(self, task_dest, "Room not found")[0], getattr(self, task_dest, "Room not found")[1])
                    self.speak("Going to wait at the " + task_dest)
                elif task_type == "grab":
                    print("GOING FOR GRAB")
                    self.speak("Grabbing the " + task_dest)
                    self.grab(task_dest)
                    print("DONE WITH GRAB")
                elif task_type == "follow":
                    self.follow_to(getattr(self, task_dest, "Room not found")[0], getattr(self, task_dest, "Room not found")[1])
            else:
                time.sleep(0.1)

    def grab(self, target):
        print("TIMEOUT")
        time.sleep(5)
        print("TIMEOUT DONE")
        self.update_action()
        return None

    # Base navigation function for map navigation after map load
    def navigate_to(self, x, y, theta=0.0):
        # self.play_music_with_fade_in(self.music_file)
        self.is_navigating = True

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.z = np.sin(theta / 2.0)
        goal_msg.pose.pose.orientation.w = np.cos(theta / 2.0)

        self.nav_client.send_goal_async(goal_msg).add_done_callback(self.goal_response_callback)

# THE BELOW SECTION IS USED FOR PERSON FOLLOWING WHICH IS CURRENTLY DISABLED.

    #Base follow function with human detection and distance calculation
    # def follow_to(self, x, y):
    #     self.is_navigating = True
    #     # self.play_music_with_fade_in(self.music_file)
    #     self.get_logger().info(f"Following to destination at ({x}, {y})")

    #     follow_goal_threshold = 0.25  
    #     original_goal_threshold = self.goal_threshold
    #     self.goal_threshold = follow_goal_threshold

    #     timeout = 30  
    #     start_time = time.time()

    #     while self.is_navigating:
    #         if time.time() - start_time > timeout:
    #             # self.fade_out_and_stop_music()
    #             self.get_logger().info("Timeout reached while following. Stopping follow.")
    #             self.speak(f"I could not reach the {self.destination} in time.")
    #             self.is_navigating = False
    #             self.stop_movement()
    #             if self.task_queue.empty():
    #                 self.speak("Any further commands?")
    #             break

    #         if self.reached_goal(x, y):
    #             # self.fade_out_and_stop_music()
    #             self.get_logger().info(f"Reached the destination at ({x}, {y}) while following.")
    #             self.speak(f"I have reached the {self.destination}.")
    #             self.is_navigating = False
    #             self.stop_movement()
    #             break

    #         person_detected, person_position = self.detect_person()
    #         if person_detected:
    #             self.adjust_velocity_to_follow(person_position)
    #             self.twist.angular.z = 0.0  
    #         else:
    #             self.get_logger().info("Lost track of the person. Stopping and scanning.")
    #             self.stop_movement()
    #             self.perform_camera_scan() 

    #         time.sleep(0.1)

    #     self.goal_threshold = original_goal_threshold

    # #Perofrm idle rotation in a 90 degree arch to locate person
    # def perform_camera_scan(self):
    #     self.speak("Lost track of person, performing scan.")
    #     # self.play_music_with_fade_in(self.waiting_music)
    #     scan_speed = 0.2
    #     scan_range = math.pi / 4  
    #     scan_duration = scan_range / scan_speed 

    #     start_time = time.time()
    #     direction = 1

    #     point = 0
    #     prevpoint = -1

    #     while time.time() - start_time < scan_duration * 2:
    #         self.twist.angular.z = direction * scan_speed
    #         self.publisher_.publish(self.twist)

    #         if time.time() - start_time >= scan_duration:
    #             if point == 0 and prevpoint == -1:
    #                 point = 1
    #                 prevpoint = 0
    #                 start_time = time.time()
    #             elif point == 0 and prevpoint == 1:
    #                 point = -1
    #                 prevpoint = 0
    #                 start_time = time.time()
    #             elif point == 1 and prevpoint == 0:
    #                 point = 0
    #                 prevpoint = 1
    #                 start_time = time.time()
    #             elif point == -1 and prevpoint == 0:
    #                 point = 0
    #                 prevpoint = -1
    #                 start_time = time.time()

    #             if point == -1 or point == 1:
    #                 direction *= -1
    #                 start_time = time.time()

    #         person_detected, person_position = self.detect_person()
    #         if person_detected:
    #             self.get_logger().info("Person detected during scan. Aligning camera forward.")
    #             # self.fade_out_and_stop_music()
    #             self.twist.angular.z = 0.0
    #             self.publisher_.publish(self.twist)
    #             return

    #         time.sleep(0.1)

    #     self.twist.angular.z = 0.0
    #     self.publisher_.publish(self.twist)

    #OpenCV upper body person detection 
    # def detect_person(self):
    #     if self.current_frame is None:
    #         return False, None

    #     gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
    #     gray = cv2.GaussianBlur(gray, (5, 5), 0)
    #     gray = cv2.equalizeHist(gray)

    #     upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml")

    #     persons = upper_body_cascade.detectMultiScale(
    #         gray,
    #         scaleFactor=1.05,
    #         minNeighbors=3,
    #         minSize=(100, 100)
    #     )

    #     largest_person = None
    #     max_area = 0
    #     for (x, y, w, h) in persons:
    #         area = w * h
    #         if area > max_area:
    #             max_area = area
    #             largest_person = (x, y, w, h)

    #     if largest_person:
    #         x, y, w, h = largest_person
    #         center_x = x + w / 2
    #         angular_offset = (center_x - self.current_frame.shape[1] / 2) / self.current_frame.shape[1]
    #         return True, {"angular_offset": angular_offset, "width": w, "height": h}

    #     return False, None

    #Follow person based on frame information, goal is to increase bounding box size of the detected person to a certain value
    # def adjust_velocity_to_follow(self, person_position):
    #     desired_box_size = 900  
    #     box_width = person_position["width"]
    #     box_height = person_position["height"]
    #     angular_offset = person_position["angular_offset"]

    #     center_threshold = 0.05

    #     if box_width < desired_box_size or box_height < desired_box_size:
    #         self.get_logger().info("Box too small, moving closer.")
    #         self.twist.linear.x = 0.5  
    #     else:
    #         self.twist.linear.x = 0.0  

    #     if abs(angular_offset) > center_threshold:
    #         self.twist.angular.z = angular_offset * -15.0  
    #         self.get_logger().info(f"Turning: angular offset {angular_offset}")
    #     else:
    #         self.twist.angular.z = 0.0  

    #     self.publisher_.publish(self.twist)

    # Movement halt
    def stop_movement(self):
        # self.fade_out_and_stop_music()
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.publisher_.publish(self.twist)

    # Updating current position based on odometry(currentley does not use map data)
    def update_current_pose(self, msg):
        self.current_pose = msg.pose.pose

    # Check for reached goal via odometry current pose value
    def reached_goal(self, x, y):
        if self.current_pose is None:
            return False

        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y

        distance_to_goal = sqrt((current_x - x) ** 2 + (current_y - y) ** 2)
        self.get_logger().info(f"Distance to goal: {distance_to_goal:.2f} meters")

        return distance_to_goal < self.goal_threshold
    
    # Pygame music enabler with fade in
    def play_music_with_fade_in(self, music_file):
        try:
            pygame.mixer.music.load(music_file)
            pygame.mixer.music.set_volume(0) 
            pygame.mixer.music.play(-1)  
            for volume in range(0, int(self.music_volume * 100) + 1, 5):
                pygame.mixer.music.set_volume(volume / 100.0)
                time.sleep(0.1) 
        except Exception as e:
            self.get_logger().error(f"Error during music fade-in for {music_file}: {e}")

    # Pygame muisc half and fadeout
    def fade_out_and_stop_music(self):
        try:
            for volume in range(int(self.music_volume * 100), -1, -5):
                pygame.mixer.music.set_volume(volume / 100.0)
                time.sleep(0.1)  
            pygame.mixer.music.stop()
        except Exception as e:
            self.get_logger().error(f"Error during music fade-out: {e}")

    # Map navigation goal callback regarding if goal is reachable
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal was rejected.')
            self.speak("Navigation goal was rejected.")
            self.is_navigating = False
            self.update_action()
            return

        self.current_goal_handle = goal_handle
        self.get_logger().info('Goal accepted, waiting for result...')
        goal_handle.get_result_async().add_done_callback(self.get_result_callback)

    # Map navigation goal result callback regarding if the robot reached the goal or not
    def get_result_callback(self, future):
        result = future.result()
        if result is not None and hasattr(result, 'status') and result.status == 4:
            # self.fade_out_and_stop_music()
            self.get_logger().info('Navigation goal succeeded!')
            self.speak(f"I have reached the {self.destination}.")
            if self.current_task == "wait":
                # self.play_music_with_fade_in(self.waiting_music)
                self.speak("Waiting for 5 seconds")
                time.sleep(5)
                self.get_logger().info("Finished waiting at the destination.")
                # self.fade_out_and_stop_music()
                self.speak(f"Finished waiting.")
            if self.task_queue.empty():
                self.speak("Any further commands?")
        else:
            self.get_logger().info('Navigation goal failed.')
            self.speak(f"I could not reach the {self.destination}.")
        self.update_action()
        self.is_navigating = False
        self.current_goal_handle = None

    # Audio output enabler
    def speak(self, text):
        with self.speak_lock:
            self.speaking = True
            try:
                self.get_logger().info(f'Speaking: {text}')
                self.text_to_speech(text)
                time.sleep(0.2)
            finally:
                self.speaking = False

    # Audio feedback enabler
    def text_to_speech(self, text):
        try:
            mp3_fp = BytesIO()
            tts = gTTS(text)
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)

            audio = AudioSegment.from_file(mp3_fp, format="mp3")
            
            audio = audio.set_channels(1).set_frame_rate(22050).set_sample_width(2)
            raw = audio.raw_data

            play = sa.play_buffer(raw, num_channels=1, bytes_per_sample=2, sample_rate=22050)
            play.wait_done()

            del audio
            mp3_fp.close()

            # with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio_file:
            #     temp_audio_path = temp_audio_file.name
            #     tts.save(temp_audio_path)
            # playsound.playsound(temp_audio_path)
            # os.remove(temp_audio_path)

        except Exception as e:
            self.get_logger().error(f'Error during TTS or playback: {e}')


# App initializer.
def create_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/actions")
    def get_actions():
        with actions_lock:
            return ACTIONS
    
    class QueueUpdate(BaseModel):
        order: list[int]

    @app.post("/update_queue")
    def update_queue(update: QueueUpdate):
        if ROS_NODE is None:
            return {"status": "error", "reason": "ROS node not ready"}
        ROS_NODE.reorder_tasks(update.order)
        return {"status": "ok"}

    return app

# Starts http server for the companion app with get and post methods.
def start_http_server():
    app = create_app()
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    server.run()

def main(args=None):
    global ROS_NODE
    rclpy.init(args=args)
    node = VoiceControlNode()
    ROS_NODE = node

    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()
    node.get_logger().info("HTTP /actions server started on 0.0.0.0:8000")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
