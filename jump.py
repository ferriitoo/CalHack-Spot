from bosdyn.client import spot
from bosdyn.client.spot import robot_command_pb2
from bosdyn.client.spot import robot_command
import bosdyn.client

# Create a connection to the robot
sdk = bosdyn.client.create_standard_sdk('MyApp')
robot = sdk.create_robot('192.168.80.3')
robot.authenticate('admin', '2zqa8dgw7lor')

# Parameters for the jump
jump_height = 0.5  # Set the desired jump height in meters
jump_duration = 2.0  # Set the duration of the jump in seconds

# Create a robot command client
robot_command_client = robot.ensure_client(robot_command.Client.name)

# Create a Jump command
jump_cmd = robot_command_pb2.JumpCommandParams()
jump_cmd.height = jump_height
jump_cmd.duration = jump_duration

# Send the jump command to the robot
robot_command_client.power_jump(jump_cmd)

# Disconnect from the robot
robot.power_off()
