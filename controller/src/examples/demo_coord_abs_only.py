#!/usr/bin/env python3
import numpy as np

import rospy
import tf.transformations as trans

from controller.msg import Trajectory_point, Trajectory_msg


def main():
    # starting ROS node and subscribers
    rospy.init_node("trajectory_test", anonymous=True)
    pub = rospy.Publisher("/trajectory", Trajectory_msg, queue_size=1)
    rospy.sleep(0.1)

    # --------------------------------------------------

    msg = Trajectory_msg()
    msg.header.stamp = rospy.Time.now()
    msg.mode = "absolute"
    msg.trajectory = [
        Trajectory_point(
            positionAbsolute = [0.45, 0.0, 0.2],
            orientationAbsolute = [1, 0, 0, 0],
            pointTime = 10.0),
        Trajectory_point(
            positionAbsolute = [0.45, 0.0, 0.2],
            orientationAbsolute = trans.quaternion_about_axis(np.pi, [np.cos(np.pi/8), np.sin(np.pi/8), 0]),
            pointTime = 8.0),
        Trajectory_point(
            positionAbsolute = [0.45, 0.0, 0.2],
            orientationAbsolute = [1, 0, 0, 0],
            pointTime = 8.0),
    ]
    pub.publish(msg)
    
    print("sent msg (coordinated)")
    rospy.sleep(26)


if __name__ == "__main__":
    main()


