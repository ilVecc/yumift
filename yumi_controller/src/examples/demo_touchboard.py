#!/usr/bin/env python3
import numpy as np

import rospy
import tf.transformations as trans

from yumi_controller.msg import Trajectory_point, Trajectory_msg


def main():
    # starting ROS node and subscribers
    rospy.init_node("demo_touchboard", anonymous=True)
    pub = rospy.Publisher("/trajectory", Trajectory_msg, queue_size=1)
    rospy.sleep(0.1)

    # --------------------------------------------------

    msg = Trajectory_msg()
    msg.header.stamp = rospy.Time.now()
    msg.mode = "coordinated"
    msg.trajectory = [
        Trajectory_point(
            positionAbsolute = [0.45, 0.0, 0.3],
            orientationAbsolute = [1, 0, 0, 0],
            positionRelative = [0, 0.22, 0],
            orientationRelative = trans.quaternion_about_axis(np.radians(180), [1, 0, 0]),
            pointTime = 8.0),
    ]
    pub.publish(msg)
    
    print("positioning message sent")
    rospy.sleep(8)


if __name__ == "__main__":
    main()


