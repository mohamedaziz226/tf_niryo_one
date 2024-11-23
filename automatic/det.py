import tensorflow as tf
import numpy as np
import cv2
from niryo_one_tcp_client import *
from niryo_one_camera import *
from utils import *

# Load the model
model = tf.saved_model.load("/home/roua/project/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/saved_model")

# Set up the robot connection
robot_ip_address = "10.10.10.10"

def get_button_color(frame, box):
    """
    Extract the region from the frame and return the dominant color
    in the area of the button.
    """
    ymin, xmin, ymax, xmax = box
    button_region = frame[int(ymin * frame.shape[0]):int(ymax * frame.shape[0]),
                          int(xmin * frame.shape[1]):int(xmax * frame.shape[1])]
    
    # Convert the button region to HSV color space
    hsv = cv2.cvtColor(button_region, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for red, green, and blue in HSV
    color_ranges = {
        'red': [(0, 120, 70), (10, 255, 255)],  # Lower and upper bounds for red
        'green': [(35, 50, 50), (85, 255, 255)],  # Lower and upper bounds for green
        'blue': [(100, 50, 50), (140, 255, 255)]  # Lower and upper bounds for blue
    }
    
    # For each color, apply a mask and count the number of pixels
    color_counts = {}
    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        color_counts[color] = np.sum(mask)  # Count the number of pixels of the color

    # Find the color with the highest count
    dominant_color = max(color_counts, key=color_counts.get)
    
    return dominant_color

def main():
    # Connect to the robot
    robot = NiryoOneClient()
    robot.connect(robot_ip_address)

    try:
        print("Press 'q' to quit.")
        while True:
            # Get a frame from the robot's camera
            succ,img=take_workspace_img(robot)
            
            # Resize the frame to the model's input size
            frame_resized = cv2.resize(img, (640, 640))
            input_tensor = tf.convert_to_tensor(frame_resized)
            input_tensor = input_tensor[tf.newaxis, ...]

            # Run the model
            detections = model(input_tensor)

            # Extracting detection details
            boxes = detections['detection_boxes']
            class_ids = detections['detection_classes']
            scores = detections['detection_scores']

            # Process detections
            for i in range(boxes.shape[0]):  # Loop through the detections
                for j in range(boxes.shape[1]):  # Loop through each detection
                    score = scores[0][j].numpy()  # Get score as a scalar value
                    class_id = int(class_ids[0][j].numpy())
                    
                    # Assuming class_id corresponds to a button-like object in the model (like a "button" or similar object)
                    if score > 0.5:  # Confidence threshold
                        box = boxes[0][j].numpy()
                        ymin, xmin, ymax, xmax = box
                        
                        # Draw bounding box on the frame
                        cv2.rectangle(img, (int(xmin * img.shape[1]), int(ymin * img.shape[0])),
                                      (int(xmax * img.shape[1]), int(ymax * img.shape[0])), (0, 255, 0), 2)
                        
                        # Get the dominant color of the button
                        button_color = get_button_color(frame_resized, box)
                        
                        # Display the class, score, and color of the button
                        cv2.putText(frame_resized, f"Color: {button_color}, Score: {score:.2f}",
                                    (int(xmin * frame_resized.shape[1]), int(ymin * frame_resized.shape[0]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Show the frame with the bounding boxes and color labels
            cv2.imshow("Button Detection", frame_resized)

            # Break the loop if the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        # Disconnect from the robot
        robot.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

