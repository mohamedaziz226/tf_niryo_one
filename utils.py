import cv2
import numpy as np
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

# Fonction pour créer un masque basé sur HSV
def create_hsv_mask(img, hsv_lower, hsv_upper):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, hsv_lower, hsv_upper)
    return mask

# Remplir les trous dans un masque
def fill_holes(mask):
    im_floodfill = mask.copy()
    h, w = mask.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, flood_mask, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    filled_mask = mask | im_floodfill_inv
    return filled_mask

# Extraire les objets détectés dans un masque
def extract_objs(img, mask, min_size=64):
    # Trouver les contours selon la version de OpenCV
    contours, *rest = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objs = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= min_size and h >= min_size:
            obj_img = img[y:y + h, x:x + w]
            objs.append((x, y, w, h, obj_img))
    return objs

# Fonction principale pour détecter les boutons et les trous
def process_frame(frame):
    # Définir les plages HSV pour les boutons et trous
    hsv_black_lower = np.array([0, 0, 0])
    hsv_black_upper = np.array([180, 255, 50])

    hsv_green_lower = np.array([40, 50, 50])
    hsv_green_upper = np.array([80, 255, 255])

    hsv_red_lower1 = np.array([0, 50, 50])
    hsv_red_upper1 = np.array([10, 255, 255])
    hsv_red_lower2 = np.array([170, 50, 50])
    hsv_red_upper2 = np.array([180, 255, 255])

    hsv_hole_lower = np.array([100, 50, 50])
    hsv_hole_upper = np.array([140, 255, 255])

    # Créer les masques pour chaque catégorie
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    black_mask = create_hsv_mask(hsv_img, hsv_black_lower, hsv_black_upper)
    green_mask = create_hsv_mask(hsv_img, hsv_green_lower, hsv_green_upper)
    red_mask1 = create_hsv_mask(hsv_img, hsv_red_lower1, hsv_red_upper1)
    red_mask2 = create_hsv_mask(hsv_img, hsv_red_lower2, hsv_red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    hole_mask = create_hsv_mask(hsv_img, hsv_hole_lower, hsv_hole_upper)

    # Nettoyer les masques
    black_mask = fill_holes(black_mask)
    green_mask = fill_holes(green_mask)
    red_mask = fill_holes(red_mask)
    hole_mask = fill_holes(hole_mask)

    # Extraire les objets
    black_buttons = extract_objs(frame, black_mask)
    green_buttons = extract_objs(frame, green_mask)
    red_buttons = extract_objs(frame, red_mask)
    holes = extract_objs(frame, hole_mask)

    # Annoter l'image
    for x, y, w, h, obj_img in black_buttons:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
        cv2.putText(frame, "Black Button", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    for x, y, w, h, obj_img in green_buttons:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Green Button", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    for x, y, w, h, obj_img in red_buttons:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "Red Button", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    for x, y, w, h, obj_img in holes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Hole", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return frame
robot_ip = '10.10.10.10'
stream_url = f'rtsp://{robot_ip}/stream'  # Utilisez la f-string correctement

def main():
    # Connecter la caméra du robot
    cap = cv2.VideoCapture(stream_url)  # Utilisez la variable f-string ici
    if not cap.isOpened():
        print(f"Erreur : Impossible d'accéder à la caméra du robot à l'URL {stream_url}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Largeur
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Hauteur
    cap.set(cv2.CAP_PROP_FPS, 30)  # Taux de rafraîchissement

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur : Impossible de lire une image.")
            break

        # Traiter l'image pour détecter boutons et trous
        processed_frame = process_frame(frame)

        # Afficher l'image annotée
        cv2.imshow("Detection", processed_frame)

        # Quitter en appuyant sur 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
