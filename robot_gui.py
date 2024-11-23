import numpy as np
import cv2
import os
import math
import ctypes
import time
import copy
import threading
import pygame
import pygame_menu
import tensorflow as tf
from niryo_one_tcp_client import *
from niryo_one_camera import *
from PIL import Image, ImageTk
import tkinter as tk

import utils
import labelling
import training

# --- Robot connection parameters ---
robot_ip_address = "192.168.0.63"  # Change this to your robot's IP address
workspace = "workspace_1"  # Name of your workspace
observation_pose = PoseObject(
    x=0.20, y=0, z=0.4,
    roll=0.0, pitch=math.pi / 2 + 0.05, yaw=0.0,
)

drop_pose = PoseObject(
    x=0.20, y=0.20, z=0.10,
    roll=0.0, pitch=math.pi / 2, yaw=0.0,
)

sleep_joints = [0.0, 0.55, -1.2, 0.0, 0.0, 0.0]
z_offset = -0.00
model = None

# --- Initialize NiryoOne client ---
client = NiryoOneClient()
client.connect(robot_ip_address)
client.calibrate(CalibrateMode.AUTO)
client.change_tool(RobotTool.GRIPPER_2)
client.move_pose(*observation_pose.to_list())

# --- Initialize pygame for GUI ---
font = cv2.FONT_HERSHEY_SIMPLEX
pygame.init()
logo = None
logo_big = None
try:
    logo = pygame.image.load('Niryo_logo/logo.png')
    logo_big = pygame.image.load('Niryo_logo/logo_big.png')
    pygame.display.set_icon(logo)
except:
    pass

if os.name == 'nt':
    ctypes.windll.user32.SetProcessDPIAware()
    window_x, window_y = (ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1))
    surface = pygame.display.set_mode((window_x, window_y), pygame.NOFRAME | pygame.FULLSCREEN)
else:
    infoObject = pygame.display.Info()
    window_x, window_y = infoObject.current_w, infoObject.current_h
    surface = pygame.display.set_mode((window_x, window_y), pygame.NOFRAME)
font_size = int(window_y / 20)
pygame.display.set_caption("Tensorflow & Niryo One")

# --- Function to draw the background ---
def draw_background(img):
    surface.fill((0x20, 0x35, 0x67))
    if logo_big is not None:
        surface.blit(logo_big, ((window_x - window_y - logo_big.get_size()[0]) / 2, window_y / 8))
    img = cv2.resize(img, (window_y, window_y))
    img = np.flip(img[:][:])  # BGR to RGB
    img = np.rot90(img, 1, (1, 0))
    img = np.flip(img, 0)
    surf = pygame.surfarray.make_surface(img)
    surface.blit(surf, (window_x - window_y, 0))
    cv2.destroyAllWindows()

# --- Function to get objects from the workspace ---
def get_all_objs():
    a, img_work = utils.take_workspace_img(client)
    img_work = utils.standardize_img(img_work)
    if not a:
        a, img_work = debug_markers(img_work)
        return img_work, None
    mask = utils.objs_mask(img_work)
    objs = utils.extract_objs(img_work, mask)
    if len(objs) == 0:
        return img_work, []
    imgs = []
    if model is None:
        return img_work, objs
    for x in range(len(objs)):
        imgs.append(cv2.resize(objs[x].img, (64, 64)))
    imgs = np.array(imgs)
    predictions = model.predict(imgs)
    for x in range(len(predictions)):
        obj = objs[x]
        obj.type = predictions[x].argmax()
        # graphical debug
        cv2.drawContours(img_work, [obj.box], 0, (0, 0, 255), 2)
        pos = [obj.square[0][1], obj.square[1][1]]
        img_work = cv2.putText(img_work, objects_names[obj.type], tuple(pos), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        pos[0] += img_work.shape[0]
        img_work = cv2.putText(img_work, objects_names[obj.type], tuple(pos), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return img_work, objs

# --- Function to pick object by name ---
def pick_by_name(name):
    if model is None:
        return
    img_work, objs = get_all_objs()
    if objs is None:
        return
    shape = img_work.shape
    for x in range(len(objs)):
        if objects_names[objs[x].type] == name:
            print("object find")
            a, obj = client.get_target_pose_from_rel(workspace, z_offset, objs[x].x / shape[0], objs[x].y / shape[1], objs[x].angle)
            client.pick_from_pose(*obj.to_list())
            client.place_from_pose(*drop_pose.to_list())
            break
    client.move_pose(*observation_pose.to_list())

# --- Function to close the application ---
def close_aplication():
    client.move_joints(*sleep_joints)
    client.set_learning_mode(True)
    exit(0)

# --- Main GUI (Tkinter and Pygame) ---
class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Interface de Commande Niryo One")
        self.geometry("600x500")
        self.configure(bg='white')

        # Initialize Niryo client
        self.connect_robot()

        # Load logo
        self.charger_logo()

        # Track button color state
        self.boutons_colors_selected = [False] * 4
        self.couleurs = ["", "Vert", "Rouge", "Noir", "vide"]

        # Main frame for buttons
        self.frame = tk.Frame(self, bg='white')
        self.frame.pack(pady=20)

        # Instructions and button creation
        self.ajouter_instructions()
        self.creer_boutons()

        # Add 'Start' button
        self.ajouter_bouton_lancer()

    def connect_robot(self):
        """Connect to the Niryo One robot."""
        try:
            self.client = NiryoOneClient()
            self.client.connect(robot_ip_address)
            self.client.calibrate(CalibrateMode.AUTO)
            self.client.change_tool(RobotTool.GRIPPER_2)
            print("Connected to Niryo One.")
        except Exception as e:
            print(f"Error connecting to robot: {e}")

    def charger_logo(self):
        """Load and display the logo."""
        try:
            self.logo_image = Image.open(r'C:\Users\utilisateur\Desktop\interface\interface\img1.jpg')  # Adjust path
            self.logo_image = self.logo_image.resize((100, 100), Image.LANCZOS)
            self.logo = ImageTk.PhotoImage(self.logo_image)
            logo_label = tk.Label(self, image=self.logo, bg='white')
            logo_label.pack(side=tk.TOP, anchor='nw', padx=10, pady=10)
        except Exception as e:
            print(f"Error loading logo: {e}")

    def ajouter_instructions(self):
        """Add instructions for the user."""
        label_instructions = tk.Label(self.frame, text="Select your command:", bg='white', font=('Arial', 14))
        label_instructions.grid(row=0, column=0, columnspan=2, pady=10)

    def creer_boutons(self):
        """Create four buttons with dropdown menus."""
        for i in range(4):
            self.creer_bouton_avec_menu(i)

    def creer_bouton_avec_menu(self, i):
        """Create a button with a dropdown menu."""
        sous_frame = tk.Frame(self.frame, bg='white')
        sous_frame.grid(row=i + 1, column=0, padx=10, pady=5, sticky="w")
        bouton = tk.Button(sous_frame, text=f"Bouton {i + 1}", bg='white', width=20)
        bouton.grid(row=0, column=0, padx=10)
        bouton.config(command=lambda idx=i: self.pick_and_place_action(idx))

        couleur_selectionnee = tk.StringVar(self)
        couleur_selectionnee.set(self.couleurs[0])

        menu_couleur = tk.OptionMenu(sous_frame, couleur_selectionnee, *self.couleurs,
                                     command=lambda couleur, b=bouton, idx=i: self.changer_couleur(b, couleur, idx))
        menu_couleur.grid(row=0, column=1)

    def changer_couleur(self, bouton, couleur, index):
        """Change the color of the selected button."""
        if couleur == "vide":
            bouton.config(bg="SystemButtonFace")
            self.boutons_colors_selected[index] = False
        elif couleur == "":
            bouton.config(bg="white")
            self.boutons_colors_selected[index] = False
        else:
            bouton.config(bg=couleur)
            self.boutons_colors_selected[index] = True

    def ajouter_bouton_lancer(self):
        """Add a 'Launch' button."""
        bouton_lancer = tk.Button(self, text="Launch", bg="#921210", fg="white", command=self.action_lancer)
        bouton_lancer.pack(side=tk.BOTTOM, anchor='se', padx=20, pady=20)

    def pick_and_place_action(self, button_index):
        """Trigger pick and place action for selected button."""
        threading.Thread(target=self.execute_robot_action, args=(button_index,)).start()

    def execute_robot_action(self, button_index):
        """Execute the pick and place logic for the robot."""
        try:
            print(f"Executing action for button {button_index + 1}...")
            self.client.move_joints(0.0, 0.5, -1.0, 0.0, 0.0, 0.0)  # Example position for observation
            # Add logic for picking and placing specific objects
            print(f"Action completed for button {button_index + 1}.")
        except Exception as e:
            print(f"Error during execution: {e}")

    def action_lancer(self):
        """Action triggered when the 'Launch' button is clicked."""
        print("Command launched!")
        threading.Thread(target=self.execute_robot_action_sequence).start()

    def execute_robot_action_sequence(self):
        """Execute a full pick and place sequence."""
        try:
            print("Starting pick-and-place sequence...")
            # Add the full sequence of robot movements
            self.client.move_joints(0.0, 0.5, -1.0, 0.0, 0.0, 0.0)
            time.sleep(1)
            print("Sequence completed.")
        except Exception as e:
            print(f"Error in sequence: {e}")

if __name__ == "__main__":
    app = Application()
    app.mainloop()
