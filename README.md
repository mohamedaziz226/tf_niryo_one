## Introduction

This demonstrator uses TensorFlow, an open-source machine learning tool developed by Google, and the MobileNetV2 model, to enable the Niryo One robot to recognize multiple objects on its workspace. This project is specifically designed for the pick-and-place application of buttons.

The system utilizes an image classification model based on MobileNetV2, a lightweight and efficient pre-trained neural network. With this model and the integration of the robot's Vision Set, artificial intelligence, image processing, and machine learning, the robot can identify buttons based on their appearance.

The MobileNetV2 model is trained on images of buttons (e.g., buttons of different shapes, sizes, and colors) to allow the robot to pick and place these buttons with great precision, based on the model's predictions.

The Niryo One robot can then identify the position of the buttons on its workspace and perform pick-and-place actions according to the detected object's location.

## Requirements

A Niryo One robot,
A large gripper (Large Gripper),
The Vision Set and its workspace (used to capture images and detect objects on the workspace),
Various buttons to be placed on the workspace (e.g., buttons of different shapes, sizes, and colors).

## Installation

#### Hardware

Workspace Calibration
Calibrating the Workspace:
Start by calibrating the workspace using Niryo One Studio. The workspace name must match the one used in the robot’s program (by default, the name is "workspace_1").

Adjusting the Observation Pose:
If your workspace is not directly in front of the robot, you will need to adjust the "observation_pose" variable in the program. This will allow the robot to see the four markers placed on the workspace.

Stability of the Robot and Workspace:
It is crucial to securely attach both the robot and the workspace to prevent any movement that could affect precision during operations.

Automatic Learning Mode:
When the program starts, if the robot cannot see the four markers from its observation pose, it will automatically switch to learning mode. The graphical interface will turn red.

Adjusting the Camera Position:
If the robot cannot see the markers, you will need to manually adjust the camera position so that the robot can detect them. Once the four markers are visible, the interface will turn green.

Confirming the Position:
After the markers are visible, you can click on the screen to confirm the current position. This pose will be saved for future use, but you can always modify it from the "Observation pose" settings menu.

#### Software

Installation de l'API Python TCP de Niryo One
ownload the Niryo One Python TCP API:
Download the Niryo One Python TCP API from the following link: [Download the Niryo One TCP API]

Access the Documentation:
Once the API is downloaded, go to the "Documentation" tab and click on "Python TCP API Documentation" to get detailed information on installation and usage.

Modify the Variables in the Code:
Open the robot_gui.py file and modify the following variables to match your setup:

robot_ip_address: the IP address of your robot.
workspace: the name of the workspace used for the robot's manipulations.
Example of modification in the code:

python
Copier le code
robot_ip_address = "IP address of your robot"
workspace = "Workspace name of your robot"
This allows the program to connect to the Niryo One robot over the network and assign a workspace for the robot's manipulations.

##### On Windows.
You must start by installing Anaconda to use the demonstrator’s installation script.

Anaconda must be installed on its default location (C:\Users\<user_name>\anaconda3).

You will find Anaconda’s installation link down below: 

https://docs.anaconda.com/anaconda/install/


Two solutions are available: 
###### Simplified installation
In the demonstrator’s folder: 

1. Launch setup.bat to install all the used libraries
2. Accept the installation of these libraries
3. Launch run.bat to launch the program

The program should launch. If it doesn’t, launch a manual installation.

###### Manual installation
1. Open a terminal from Anaconda Navigator (CMD.exe Prompt, “Launch”)

You should see “(base)” displayed to the left of your terminal.

2. Update Anaconda
```python
conda update -n base -c defaults conda
```

3. Create a TensorFlow 2 environment with python 3.6
```python
conda create -n tf_niryo_one tensorflow=2 python=3.6
```

4. Enable TensorFlow’s environment
```python
conda activate tf_niryo_one
```

You should now see “(tf_niryo_one)” instead of “(base)” on the left of your terminal.

5. Update Tensor Flow
```python
pip install --upgrade tensorflow
```

6. Install opencv, pygame and pygame-menu libraries
```python
install opencv-python pygame pygame-menu
```

7. Get in the demonstrator’s folder
```python
cd Desktop\tf_niryo_one-master
```

8. Launch the program
```python
python robot_gui.py
```


##### On Linux
1. Install Anaconda
https://docs.anaconda.com/anaconda/install/

2. Open a terminal

You should find “(base)” displayed on the left of your username.

3. Update Anaconda
```python
conda update -n base -c defaults conda
```

4. Create a TensorFlow 2 environment with python 3.6
```python
conda create -n tf_niryo_one tensorflow=2 python=3.6
```

5. Enable TensorFlow’s environment
```python
conda activate tf_niryo_one
```

You should now see “(tf_niryo_one)” instead of “(base)” on the left of your terminal.

6. Update TensorFlow
```python
pip install --upgrade tensorflow
```

7. Install opencv, pygame and pygame-menu libraries
```python
pip install opencv-python pygame pygame-menu
```

8. Get in the demonstrator’s folder
```python
cd tensorflow_niryo_one/
```

9. Launch the program
```python
python robot_gui.py
```

## Functioning 

#### Creation of the database(labelling.py)

To create your database, you need to take pictures of the objects you want to use. Take at least 20 pictures of each object to get good results.

The aim is to take pictures of each object under a multitude of angles and different lighting conditions. The pictures will be stored in a folder named with the name of the concerned object, inside the “data” folder.

#### Tracking of the objects (utils.py)

##### Image shooting (“take_workspace_img()” function)

Use the TCP API to ask the robot to send an image, to crop it and to compensate the lens’ distortion.



##### Objects’ extraction (“[extract_objs()](https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#gadf1ad6a0b82947fa1fe3c3d497f260e0 "extract_objs()")” function)

Use [cv2.findContours()](https://docs.opencv.org/master/d4/d73/tutorial_py_contours_begin.html "cv2.findContours()") to obtain the list of the outline of the objects being on the previously calculated mask.


With [cv2.minAreaRect()](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.htmlhttp:// "cv2.minAreaRect()") we obtain a square containing the smallest object and use this information to extract the object from the image and put it vertically (giving the same orientation to these images makes the recognition easier for TensorFlow).


#### Training (training.py)

Launch training.py or click on the “Train” button on the graphic interface. This creates a TensorFlow model (neural network). Then create a list which contains all the images from the “data” folder and a list which contains the label corresponding to each image.

Use [`<modele>`.fit(`<images>,` `<labels>`)](https://www.tensorflow.org/api_docs/python/tf/keras/Model "<modele>.fit(<images>, <labels>)") to train the model with the database.

When the training is over, test the model’s performances and save it in the “model” folder.

#### Prediction (robot.py / robot_gui.py)

Launch robot.py or the graphic interface’s “lancer” menu:

Use the previously trained model to recognize the different objects on the workspace.



#### OTHER FEATURES 

- Replace the images in the “logo” folder with customized logos (black is used as a transparency color).

- Adds or removes images and folders in the database from a file management tool (use the “Update” button to ask the application to rescan the folders)

- Provided data sets: 

	- Two data sets based on the Celebration chocolates.
	- A data set with 212 images which allows you to train a model with 85 / 90% accuracy (1 to 3 minutes of training)
	- A data set with 963 images which allows you to train a model with 95 / 99.5% accuracy (1 to 15 minutes of training)

##  Launching the program

Launch “run.bat” (Windows only) or enter the command “python3 robot_gui.py”..

/ ! \ Assure you to be in an environment having TensorFlow as well as the necessary Python libraries (“conda activate tf_niryo_one”).

```python
(base) Users\user> conda activate tf_niryo_one
(tf_niryo_one) Users\user>
```

