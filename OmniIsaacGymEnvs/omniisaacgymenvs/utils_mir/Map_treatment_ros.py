#!/usr/bin/env python3

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import rospy
from nav_msgs.msg import Odometry
import threading  # Import the threading module
#from geometry_msgs import PoseWithCovariance, TwistWithCovariance

image = Image.open('/home/vtls/Documents/Thesis/Occupancy_map.png')
image = image.convert('L')
image_data = np.array(image)
binary_array = (image_data > 50).astype(int)



def add_toMap(x,y,size,clasif):
    binary_array = (image_data > 50).astype(int)
    x_r, y_r = coord_toImg(y,x)
    for x in range(x_r-size, x_r+size):
        for y in range(y_r-size, y_r+size):
            if 0 <= x < binary_array.shape[0] and 0 <= y < binary_array.shape[1]:
                binary_array[x, y] = clasif

    return binary_array

def coord_toImg(x,y):
    x = round(x,1)
    y = round(y,1)
    x_translation = 150
    y_translation = 110
    x_scale = -10
    y_scale = 100 - 110
    
    x_transformed = int(x * x_scale + x_translation)
    y_transformed = int(y * y_scale + y_translation)

    return x_transformed,y_transformed


##################ROS###################

def callback(data):
    global binary_array

    x_r = data.pose.pose.position.x
    y_r = data.pose.pose.position.y

    # Update the occupancy array
    binary_array = add_toMap(x_r, y_r, 5, 2)


def plot_occupancy_map():
    while not rospy.is_shutdown():
        plt.clf()  # Clear the previous plot
        plt.imshow(binary_array, cmap=cmap, origin='lower', norm=colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N))
        plt.colorbar(ticks=[0, 1, 2, 3, 4])
        plt.title('Live Pose of base_link')
        plt.pause(0.01)  # Pause to allow real-time updates




if  __name__ == '__main__':

    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/odom", Odometry, callback)

    # Create a timer that calls the callback every 0.3 seconds
    timer = rospy.Timer(rospy.Duration(0.3), lambda event: None)

    # Set up the colormap
    cmap = colors.ListedColormap(['black', 'white', 'blue', 'red', 'pink'])

    # Start a separate thread for plotting
    plot_thread = threading.Thread(target=plot_occupancy_map)
    plot_thread.start()

    plt.ion()
    plt.show()

    rospy.spin()
# [0,0] is in [110,150]
