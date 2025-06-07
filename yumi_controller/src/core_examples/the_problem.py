#!/usr/bin/env python3
import numpy as np

import rospy
import tf.transformations as trans

from yumi_controller.msg import YumiPosture, YumiTrajectory


def main():
    # starting ROS node and subscribers
    rospy.init_node("demo_touchboard", anonymous=True)
    pub = rospy.Publisher("/trajectory", YumiTrajectory, queue_size=1, latch=True)
    rospy.sleep(0.1)

    # --------------------------------------------------

    msg = YumiTrajectory()
    msg.header.stamp = rospy.Time.now()
    msg.mode = "coordinated"
    msg.trajectory = [
        YumiPosture(
            positionAbsolute = [0.45, 0.0, 0.3],
            orientationAbsolute = [0, 1, 0, 0],
            positionRelative = [0, 0.22, 0],
            orientationRelative = np.roll(trans.quaternion_about_axis(np.radians(160), [1, 0, 0]), 1),  # TODO this orientation is the issue
            pointTime = 8.0),
    ]
    pub.publish(msg)
    print("positioning message sent")
    rospy.sleep(6)


if __name__ == "__main__":
    main()


