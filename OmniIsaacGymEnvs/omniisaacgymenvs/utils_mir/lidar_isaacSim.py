import rospy
from sensor_msgs.msg import LaserScan
import numpy as np
import matplotlib.pyplot as plt


def polar2carth(polar):
    LIMIT = 20 #Maximum distance in m to consider 
    angles = np.linspace(0, 2 * np.pi, len(polar)).tolist()
    carthesian_x, carthesian_y = [],[]
    for i in range(len(polar)):
        if polar[i] != float("inf"):
            if polar[i] <= LIMIT:
                carthesian_x.append(np.cos(angles[i])*polar[i])
                carthesian_y.append(np.sin(angles[i])*polar[i])

    return carthesian_x,carthesian_y


def main(polar):
    carthesian_x , carthesian_y = polar2carth(polar)
    print(carthesian_x,carthesian_y)



def lidar_callback(msg):
    main(msg.ranges)


def subscriber_node():
    rospy.init_node('lidar')
    rospy.Subscriber('/scan', LaserScan, lidar_callback)
    rospy.spin()