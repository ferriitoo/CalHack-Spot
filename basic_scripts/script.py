from bosdyn.client import Robot
from bosdyn.client.spot import Spot

# Replace these with your robot's information
# robot_ip = "192.168.50.3"
# username = "spot"
# password = "Merkleb0t"

robot_ip="192.168.50.3"
# ROBOT_IP="localhost:2000"
username = "admin"
password = "2zqa8dgw7lor"



# Create a Robot object
robot = Robot(robot_ip)
robot.authenticate(username, password)

# Create a Spot object for interacting with Spot-specific features
spot = robot.ensure_client(Spot.client_name)
