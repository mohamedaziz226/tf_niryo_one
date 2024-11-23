import cv2
import numpy as np
import tensorflow as tf
from niryo_one_tcp_client import *
from niryo_one_camera import *
from utils import *

# Charger le modèle TensorFlow
model = tf.saved_model.load("/home/roua/project/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/saved_model")

# Adresse IP du robot
robot_ip_address = "10.10.10.10"

def detect_holes(frame):
    """
    Détecte les trous dans l'image de la pièce.
    """
    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Appliquer un flou pour réduire le bruit
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Appliquer une détection des cercles avec HoughCircles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,  # Distance minimale entre les cercles
        param1=50,
        param2=30,  # Plus petit -> détection plus sensible
        minRadius=15,  # Rayon minimum des cercles
        maxRadius=35   # Rayon maximum des cercles
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            x, y, r = circle
            # Dessiner les cercles détectés
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

    return frame
def main():
    # Connexion au robot
    robot = NiryoOneClient()
    robot.connect(robot_ip_address)

    try:
        print("Press 'q' to quit.")
        while True:
            # Capturer une image à partir de la caméra du robot
            succ, img = take_workspace_img(robot)
            if not succ:
                print("Erreur : Impossible d'obtenir l'image depuis la caméra du robot.")
                continue

            # Vérifier et redimensionner l'image si nécessaire
            if img.shape[:2] != (640, 640):
                img = cv2.resize(img, (640, 640))  # Forcer la taille 640x640
                print(f"Taille de l'image ajustée : {img.shape[:2]}")

            # Redimensionner l'image pour le modèle
            frame_resized = cv2.resize(img, (640, 640))  # Assurez-vous que c'est 640x640
            input_tensor = tf.convert_to_tensor(frame_resized)
            input_tensor = input_tensor[tf.newaxis, ...]

            # Détection d'objets avec le modèle
            detections = model(input_tensor)

            # Extraire les détails des détections
            boxes = detections['detection_boxes']
            class_ids = detections['detection_classes']
            scores = detections['detection_scores']

            # Traitement des détections
            for i in range(boxes.shape[0]):
                for j in range(boxes.shape[1]):
                    score = scores[0][j].numpy()
                    class_id = int(class_ids[0][j].numpy())
                    
                    # Seuil de confiance
                    if score > 0.5:
                        box = boxes[0][j].numpy()
                        ymin, xmin, ymax, xmax = box

                        # Dessiner le rectangle de détection
                        cv2.rectangle(img, (int(xmin * img.shape[1]), int(ymin * img.shape[0])),
                                      (int(xmax * img.shape[1]), int(ymax * img.shape[0])), (0,255, 0), 2)

            # Ajouter la détection des trous
            frame_with_holes = detect_holes(img)

            # Afficher les résultats
            cv2.imshow("Robot Camera with Detection", frame_with_holes)

            # Quitter si 'q' est pressé
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        # Déconnexion du robot
        robot.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

