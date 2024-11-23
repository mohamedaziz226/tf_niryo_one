#!/usr/bin/env python

import time
import math
import cv2
import os
import numpy as np
from niryo_one_tcp_client import *
from niryo_one_camera import *
import utils  # Assurez-vous que utils.py est correctement implémenté
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2

# Paramètres
robot_ip_address = "10.10.10.10"
workspace = "workspace_1"
observation_pose = PoseObject(x=0.20, y=0, z=0.4, roll=0.0, pitch=math.pi / 2 + 0.05, yaw=0.0)
model_path = "tf_niryo_one/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/saved_model"  # Chemin vers votre modèle MobileNetV2
input_size = (224, 224) # Taille d'entrée du modèle MobileNetV2

# Chargement du modèle MobileNetV2
model = load_model("tf_niryo_one/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/saved_model")


# Fonction de détection des trous avec HSV
def detect_holes(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Définissez les plages HSV pour la couleur des trous (à ajuster selon vos besoins)
    lower_hsv = np.array([0, 0, 0])  # Exemple pour le noir
    upper_hsv = np.array([180, 255, 50]) # Exemple pour le noir
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    holes = []
    for contour in contours:
        if cv2.contourArea(contour) > 50:  # Filtrer les petits contours (bruit)
            x, y, w, h = cv2.boundingRect(contour)
            holes.append((x + w // 2, y + h // 2)) # Centre du trou
    return holes, mask

# Fonction de classification des boutons avec MobileNetV2
def classify_button(image):
    resized_image = cv2.resize(image, input_size)
    preprocessed_image = preprocess_input(resized_image)
    prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))
    # Interprétez la prédiction (ex: argmax pour la classe la plus probable)
    class_index = np.argmax(prediction)
    return class_index


def pick_and_place(client, button_position, hole_position):
    # Implémentez la logique de pick and place ici
    # Utilisez button_position et hole_position pour guider le robot
    # Exemple :
    # client.move_pose_above_target(...) # Aller au dessus du bouton
    # client.activate_vacuum_pump(...) # Activer la pompe à vide
    # client.move_pose(...) # Descendre pour prendre le bouton
    # ... # Déplacer le bouton vers le trou
    # client.deactivate_vacuum_pump(...) # Désactiver la pompe
    pass


if __name__ == "__main__":
    client = NiryoOneClient()
    client.connect(robot_ip_address)

    try:
        client.calibrate(CalibrateMode.AUTO)
        client.change_tool(RobotTool.VACUUM_PUMP_1)
    except:
        print("Calibration failed")

    client.move_pose(*observation_pose.to_list())

    while True:
        a, img_work = utils.take_workspace_img(client)

        # Détection des boutons (à adapter à vos besoins)
        button_positions, _ = detect_holes(img_work) # Vous devrez adapter cela à la détection de vos boutons

        # Détection des trous
        holes_positions, mask_holes = detect_holes(img_work)


        # Afficher l'image avec les masques (pour le débogage)
        debug_image = cv2.bitwise_or(img_work, img_work, mask=mask_holes) # Superposer le masque des trous
        show_img("Robot View", debug_image, wait_ms=50)
        

        for button_position in button_positions:
            # Rogner l'image du bouton
            x, y = button_position
            button_img = img_work[y-32:y+32, x-32:x+32] # Ajuster les valeurs de rognage si nécessaire


            # Classifier le bouton
            button_class = classify_button(button_img)

            # Trouver le trou correspondant (à adapter selon votre logique)
            # Exemple: trouver le trou le plus proche
            if holes_positions:
                 closest_hole = min(holes_positions, key=lambda hole: math.dist(button_position, hole))
                 pick_and_place(client, button_position, closest_hole)
                 holes_positions.remove(closest_hole) # Supprimer le trou utilisé


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    client.disconnect()