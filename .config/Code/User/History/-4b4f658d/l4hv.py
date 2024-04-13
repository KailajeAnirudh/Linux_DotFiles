#!/usr/bin/env python
import rosbag
#from sensor_msgs.msg import PointCloud2
import matplotlib.pyplot as plt
import numpy as np

# bag = rosbag.Bag('/home/shibos/Documents/ros_workspace/src/Measurements/02_Measurements/Hover_2024-03-11-13-39-30.bag')
#bag = rosbag.Bag('/home/shibos/Documents/ros_workspace/src/Measurements/02_Measurements/Evaluate_2024-03-11-13-56-58.bag')
#bag = rosbag.Bag('/home/shibos/Documents/ros_workspace/src/Measurements/02_Measurements/Stepz_2024-03-11-13-45-04.bag')
bag = rosbag.Bag('/home/anirudhkailaje/Documents/01_UPenn/04_MEAM6200/01_Projects/04_Project1_4/02_Measurements/Stepy_2024-03-11-13-55-44.bag')





#------------------------------------------x vs t-------------------------------------------------------------
drone_x_positions = []
flat_output_x_positions = []
timestamps_flat_x= []
timestamps_drone_x = []

for topic, msg, t in bag.read_messages(topics=['odom']):
    drone_x_positions.append(msg.pose.pose.position.x)
    timestamps_drone_x.append(msg.header.stamp.to_sec())
for topic, msg, t in bag.read_messages(topics=['flat_outputs']):
    flat_output_x_positions.append(msg.pose.position.x)
    timestamps_flat_x.append(msg.header.stamp.to_sec())

plt.figure(figsize=(10, 5))
plt.plot(timestamps_drone_x, drone_x_positions, label='Drone  Position')
plt.plot(timestamps_flat_x, flat_output_x_positions, 'r', label='Flat Output X Position')
plt.xlabel('Time (s)')
plt.ylabel('X Position')
plt.title('X Position vs. Time')
plt.grid(True)
plt.legend()
# plt.show()

#------------------------------------------y vs t-------------------------------------------------------------
drone_y_positions = []
flat_output_y_positions = []
timestamps_flat_y= []
timestamps_drone_y = []
for topic, msg, t in bag.read_messages(topics=['odom']):
    drone_y_positions.append(msg.pose.pose.position.y)
    timestamps_drone_y.append(msg.header.stamp.to_sec())
for topic, msg, t in bag.read_messages(topics=['flat_outputs']):
    flat_output_y_positions.append(msg.pose.position.y)
    timestamps_flat_y.append(msg.header.stamp.to_sec())



steady_state_time = 51
steady_state_value = np.array(drone_y_positions[np.array(timestamps_drone_y)>steady_state_time]).mean()
plt.figure(figsize=(10, 5))
plt.plot(timestamps_drone_y, drone_y_positions, label='Drone  Position')
plt.plot(timestamps_flat_y, flat_output_y_positions, 'r', label='Flat Output Y Position')
plt.hlines(steady_state_value, steady_state_time, timestamps_drone_y[-1], label = 'Steady state')
plt.xlabel('Time (s)')
plt.ylabel('Y Position')
plt.title('Y Position vs. Time')
plt.grid(True)
plt.legend()
plt.show()

#------------------------------------------z vs t-------------------------------------------------------------
drone_z_positions = []
flat_output_z_positions = []
timestamps_flat_z= []
timestamps_drone_z = []
for topic, msg, t in bag.read_messages(topics=['odom']):
    drone_z_positions.append(msg.pose.pose.position.z)
    timestamps_drone_z.append(msg.header.stamp.to_sec())
for topic, msg, t in bag.read_messages(topics=['flat_outputs']):
    flat_output_z_positions.append(msg.pose.position.z)
    timestamps_flat_z.append(msg.header.stamp.to_sec())

plt.figure(figsize=(10, 5))
plt.plot(timestamps_drone_z, drone_z_positions, label='Drone Z Position')
plt.plot(timestamps_flat_z, flat_output_z_positions, 'r', label='Falt Output Z Position')
plt.xlabel('Time (s)')
plt.ylabel('Z Position')
plt.title('Z Position vs. Time')
plt.grid(True)
plt.legend()
# plt.show()





# # Print all messages

# for topic, msg, t in bag.read_messages(topics=['flat_outputs']):
#     print(msg)
# bag.close()