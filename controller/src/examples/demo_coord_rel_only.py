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
    msg.mode = "relative"
    msg.trajectory = [
        Trajectory_point(
            positionRelative = [0, 0.30, 0],
            orientationRelative = [0, 0, 0, 1],
            pointTime = 8.0),
        Trajectory_point(
            positionRelative = [0, 0.50, 0],
            orientationRelative = trans.quaternion_about_axis(np.radians(60), [0, 1, 0]),
            pointTime = 8.0),
        Trajectory_point(
            positionRelative = [0, 0.50, 0],
            orientationRelative = trans.quaternion_about_axis(np.radians(45), [1, 0, 0]),
            pointTime = 8.0),
        Trajectory_point(
            positionRelative = [0, 0.30, 0],
            orientationRelative = [0, 0, 0, 1],
            pointTime = 8.0),
    ]
    pub.publish(msg)
    print("Sent message")
    rospy.sleep(32)


if __name__ == "__main__":
    main()


