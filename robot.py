#!/usr/bin/env python

import numpy as np
import cv2
import os
import math  # Import math for math.pi
import tensorflow as tf
from niryo_one_tcp_client import *
from niryo_one_camera import *
import utils
from tcp_client import NiryoOneClient # Si cette exception est définie dans ce module.
  # Assuming you have a utils.py file with helper functions

# --- Robot Setup ---
robot_ip_address = "10.10.10.10"  # Replace with your robot's IP address
workspace = "workspace_1"  # Name of your workspace
observation_pose = PoseObject(
    x=0.20, y=0, z=0.4,
    roll=0.0, pitch=math.pi / 2 + 0.05, yaw=0.0,
)
drop_pose = PoseObject(
    x=0.20, y=0.20, z=0.10,
    roll=0.0, pitch=math.pi / 2, yaw=0.0,
)

# --- HSV Thresholds for Hole Detection (Calibrate these!) ---
lower_hsv = np.array([0, 0, 0])   # Example: black holes
upper_hsv = np.array([179, 255, 50]) # Example: black holes



# --- Connect to Robot ---
client = NiryoOneClient()
client.connect(robot_ip_address)

# --- Calibration (if needed) ---
try:
    client.calibrate(CalibrateMode.MANUAL)
except  NiryoOneClient  as e:
    print(f"Calibration failed: {e}")

client.change_tool(RobotTool.GRIPPER_2)

# --- Main Loop ---
while True:
    try:
        # 1. Move to observation pose
        client.move_pose(*observation_pose.to_list())

        # 2. Take image (with error handling)
        img_work = None
        for _ in range(3):  # Retry a few times if image capture fails
            success, img_work = utils.take_workspace_img(client)
            if success:
                break
            else:
                print("Image capture failed. Retrying...")
        if img_work is None:
            print("Failed to capture image after multiple attempts. Skipping this iteration.")
            continue

        img_work = utils.standardize_img(img_work) # Standardize image (if necessary in utils.py)

        # 3. Hole Detection using HSV and Contours
        hsv = cv2.cvtColor(img_work, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objs = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            obj_img = img_work[y:y+h, x:x+w]
            objs.append(utils.Object(x, y, w, h, angle=0, img=obj_img))




        # 4. Display Image (for debugging)
        img_db = np.concatenate([img_work, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)], axis=1) # Combine image and mask
        for obj in objs:
            cv2.drawContours(img_db, [obj.box], 0, (0, 0, 255), 2)
            pos = [obj.square[0][1], obj.square[1][1]]
            img_db = cv2.putText(img_db, "hole", tuple(pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        img_db = utils.resize_img(img_db, width=1200, height=400) # Resize for display
        utils.show_img("Robot View", img_db, wait_ms=50)

        # 5. User Input and Object Picking
        if objs: # Check if any holes were detected
            print("Detected holes:")
            for i, _ in enumerate(objs):
                print(f"{i+1}. Hole {i+1}")  # Number the holes


            while True: # Loop until valid input is received
                string = input("Which hole do you want to pick? (Enter number): ")
                try:
                    hole_index = int(string) - 1
                    if 0 <= hole_index < len(objs):
                        selected_obj = objs[hole_index]
                        break  # Exit the input loop
                    else:
                        print("Invalid hole number. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number.")




            # Pick the selected object
            shape = img_work.shape
            success, pick_pose = client.get_target_pose_from_rel(
                workspace, -0.01, selected_obj.x / shape[0], selected_obj.y / shape[1], selected_obj.angle
            )
            if success:
                client.pick_from_pose(*pick_pose.to_list())
                client.place_from_pose(*drop_pose.to_list())
            else:
                print(f"Failed to get pick pose: {pick_pose}") # Print error if any

        else:
            print("No holes detected.")


    except NiryoOneException as e:
        print(f"Niryo One error: {e}")
        # Add error handling as needed (e.g., retry, stop the robot)