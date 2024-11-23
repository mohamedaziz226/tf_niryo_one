# --- Configuration
from niryo_one_tcp_client import *
robot_ip_address = "10.10.10.10"  # Update this with your robot's IP address
gripper_used = RobotTool.GRIPPER_2  # Using gripper 2

# --- Pose Definitions
initial_pose = PoseObject(
    x=0.216, y=0.067, z=0.382,  # Position initiale du robot
    roll=2.995, pitch=1.522, yaw=-2.380  # Orientation initiale
)

pos1 = PoseObject(
    x=-0.092, y=0.249, z=0.333,  # Position 1
    roll=-0.864, pitch=1.425, yaw=1.686
)

pos2 = PoseObject(
    x=-0.092, y=0.249, z=0.210,  # Position 2 (object to pick)
    roll=-0.590, pitch=1.470, yaw=1.936
)

pos3 = PoseObject(
    x=-0.075, y=0.197, z=0.376,  # Position 3 (after picking)
    roll=-0.708, pitch=1.232, yaw=1.842
)

pos4 = PoseObject(
    x=0.285, y=0.033, z=0.212,  # Position 4 (placing position)
    roll=-1.084, pitch=1.523, yaw=-0.335
)
pos5 = PoseObject(
    x=0.285, y=0.033, z=0.200,  # Position 5 (placing position)
    roll=-1.084, pitch=1.523, yaw=-0.335
)


# Function to adjust poses globally
def adjust_pose(pose, x_offset=0.0, y_offset=0.0, z_offset=0.0):
    """
    Adjusts a pose by applying global offsets.
    """
    pose.x += float(x_offset)  # Ensure the offsets are floats
    pose.y += float(y_offset)
    pose.z += float(z_offset)
    return pose

# Apply global adjustment if necessary
global_offsets = {'x': 0.0, 'y': 0.0, 'z': 0.0}  # Example offsets
initial_pose = adjust_pose(initial_pose, *global_offsets.values())  # Convert dict to values
pos1 = adjust_pose(pos1, *global_offsets.values())
pos2 = adjust_pose(pos2, *global_offsets.values())
pos3 = adjust_pose(pos3, *global_offsets.values())
pos4 = adjust_pose(pos4, *global_offsets.values())
pos5 = adjust_pose(pos5, *global_offsets.values())


# --- Functions
def move_to_pose(niryo_one_client, pose):
    """
    Moves the robot to a specified pose.
    """
    print(f"Moving to pose: {pose}")
    niryo_one_client.move_pose(*pose.to_list())
    print("Reached target pose.")

def open_gripper(niryo_one_client):
    """
    Opens the gripper for picking up an object.
    """
    print("Opening gripper...")
    niryo_one_client.open_gripper(gripper_used, 400)  # 400 is the speed of the gripper
    print("Gripper opened.")

def close_gripper(niryo_one_client):
    """
    Closes the gripper to grab an object.
    """
    print("Closing gripper...")
    niryo_one_client.close_gripper(gripper_used, 400)  # 400 is the speed of the gripper
    print("Gripper closed.")

# --- Main Program
if __name__ == "__main__":
    # Create and connect to the Niryo One client
    client = NiryoOneClient()
    print("Connecting to the Niryo One robot...")
    client.connect(robot_ip_address)

    # Change tool and calibrate the robot
    print("Changing tool to Gripper 2...")
    client.change_tool(gripper_used)
    
    print("Calibrating robot manually...")
    client.calibrate(CalibrateMode.MANUAL)

    # Step-by-step movement
    print("Step 1: Moving to initial position...")
    move_to_pose(client, initial_pose)

    print("Step 2: Moving to pos1...")
    move_to_pose(client, pos1)

    print("Step 3: Opening gripper...")
    open_gripper(client)

    print("Step 4: Moving to pos2 (pick position)...")
    move_to_pose(client, pos2)

    print("Step 5: Closing gripper...")
    close_gripper(client)

    print("Step 6: Moving to pos3...")
    move_to_pose(client, pos3)

    print("Step 7: Moving to pos4 (place position)...")
    move_to_pose(client, pos4)
    
    print("Step 8: Moving to pos5 (place position)...")
    move_to_pose(client, pos5)


    # Enable learning mode (optional)
    print("Enabling learning mode...")
    client.set_learning_mode(True)

    # Disconnect from the robot
    print("Disconnecting from the robot...")
    client.quit()
    print("Program finished.")# --- Configuration
from niryo_one_tcp_client import *
robot_ip_address = "10.10.10.10"  # Update this with your robot's IP address
gripper_used = RobotTool.GRIPPER_2  # Using gripper 2

# --- Pose Definitions
initial_pose = PoseObject(
    x=0.216, y=0.067, z=0.382,  # Position initiale du robot
    roll=2.995, pitch=1.522, yaw=-2.380  # Orientation initiale
)

pos1 = PoseObject(
    x=-0.092, y=0.249, z=0.333,  # Position 1
    roll=-0.864, pitch=1.425, yaw=1.686
)

pos2 = PoseObject(
    x=-0.097, y=0.261, z=0.255,  # Position 2 (object to pick)
    roll=-0.590, pitch=1.470, yaw=1.936
)

pos3 = PoseObject(
    x=-0.075, y=0.197, z=0.376,  # Position 3 (after picking)
    roll=-0.708, pitch=1.232, yaw=1.842
)

pos4 = PoseObject(
    x=0.285, y=0.033, z=0.212,  # Position 4 (placing position)
    roll=-1.084, pitch=1.523, yaw=-0.335
)

# Function to adjust poses globally
def adjust_pose(pose, x_offset=0.0, y_offset=0.0, z_offset=0.0):
    """
    Adjusts a pose by applying global offsets.
    """
    pose.x += float(x_offset)  # Ensure the offsets are floats
    pose.y += float(y_offset)
    pose.z += float(z_offset)
    return pose

# Apply global adjustment if necessary
global_offsets = {'x': 0.01, 'y': 0.0, 'z': 0.0}  # Example offsets
initial_pose = adjust_pose(initial_pose, *global_offsets.values())  # Convert dict to values
pos1 = adjust_pose(pos1, *global_offsets.values())
pos2 = adjust_pose(pos2, *global_offsets.values())
pos3 = adjust_pose(pos3, *global_offsets.values())
pos4 = adjust_pose(pos4, *global_offsets.values())

# --- Functions
def move_to_pose(niryo_one_client, pose):
    """
    Moves the robot to a specified pose.
    """
    print(f"Moving to pose: {pose}")
    niryo_one_client.move_pose(*pose.to_list())
    print("Reached target pose.")

def open_gripper(niryo_one_client):
    """
    Opens the gripper for picking up an object.
    """
    print("Opening gripper...")
    niryo_one_client.open_gripper(gripper_used, 400)  # 400 is the speed of the gripper
    print("Gripper opened.")

def close_gripper(niryo_one_client):
    """
    Closes the gripper to grab an object.
    """
    print("Closing gripper...")
    niryo_one_client.close_gripper(gripper_used, 400)  # 400 is the speed of the gripper
    print("Gripper closed.")

# --- Main Program
if name == "main":
    # Create and connect to the Niryo One client
    client = NiryoOneClient()
    print("Connecting to the Niryo One robot...")
    client.connect(robot_ip_address)

    # Change tool and calibrate the robot
    print("Changing tool to Gripper 2...")
    client.change_tool(gripper_used)
    print("Calibrating robot manually...")
    client.calibrate(CalibrateMode.MANUAL)

    # Step-by-step movement
    print("Step 1: Moving to initial position...")
    move_to_pose(client, initial_pose)

    print("Step 2: Moving to pos1...")
    move_to_pose(client, pos1)

    print("Step 3: Opening gripper...")
    open_gripper(client)

    print("Step 4: Moving to pos2 (pick position)...")
    move_to_pose(client, pos2)

    print("Step 5: Closing gripper...")
    close_gripper(client)

    print("Step 6: Moving to pos3...")
    move_to_pose(client, pos3)

    print("Step 7: Moving to pos4 (place position)...")
    move_to_pose(client, pos4)

    # Enable learning mode (optional)
    print("Enabling learning mode...")
    client.set_learning_mode(True)

    # Disconnect from the robot
    print("Disconnecting from the robot...")
    client.quit()
    print("Program finished.")
