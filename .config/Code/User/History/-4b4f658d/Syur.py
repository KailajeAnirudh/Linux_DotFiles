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

# plt.figure(figsize=(10, 5))
# plt.plot(timestamps_drone_x, drone_x_positions, label='Drone  Position')
# plt.plot(timestamps_flat_x, flat_output_x_positions, 'r', label='Flat Output X Position')
# plt.xlabel('Time (s)')
# plt.ylabel('X Position')
# plt.title('X Position vs. Time')
# plt.grid(True)
# plt.legend()
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

drone_x_positions = np.array(drone_x_positions)
drone_y_positions = np.array(drone_y_positions)
flat_output_y_positions = np.array(flat_output_y_positions)
timestamps_drone_y = np.array(timestamps_drone_y)
timestamps_flat_y = np.array(timestamps_flat_y)
timestamps_flat_y -= timestamps_flat_y[0]
timestamps_drone_y -= timestamps_drone_y[0]

steady_state_time = 5
steady_state_value = np.array(drone_y_positions)[np.array(timestamps_drone_y)>steady_state_time].mean()
max_time = timestamps_drone_y[np.argmax(drone_y_positions)]
plt.figure(figsize=(10, 5))
plt.plot(timestamps_drone_y, drone_y_positions, label='Drone  Position')
plt.plot(timestamps_flat_y, flat_output_y_positions, 'r', label='Flat Output Y Position')
plt.hlines([steady_state_value, drone_y_positions.max()], [steady_state_time, max_time], [16, max_time+5], colors = 'black', label = 'Steady state')
plt.vlines(steady_state_time, 0, 0.5, colors = 'green', linestyles='dashed')
plt.text(5, 0.59, f'Max overshoot to {drone_y_positions.max():.3f} m')
plt.text(steady_state_time+0.01, steady_state_value+0.01, f'Steady State: {steady_state_value: .3f} m, Settling time: {steady_state_time} s')
plt.xlabel('Time (s)')
plt.ylabel('Y Position (m)')
plt.title('Step Y Response')
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