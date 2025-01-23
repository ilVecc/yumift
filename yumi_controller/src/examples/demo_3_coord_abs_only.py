#!/usr/bin/env python3
import numpy as np

import rospy
import tf.transformations as trans

from yumi_controller.msg import YumiPosture, YumiTrajectory


def main():
    # starting ROS node and subscribers
    rospy.init_node("trajectory_test", anonymous=True)
    pub = rospy.Publisher("/trajectory", YumiTrajectory, queue_size=1)
    rospy.sleep(0.1)

    # --------------------------------------------------

    msg = YumiTrajectory()
    msg.header.stamp = rospy.Time.now()
    msg.mode = "absolute"
    msg.trajectory = [
        YumiPosture(
            positionAbsolute = [0.45, 0.0, 0.2],
            orientationAbsolute = [0, 1, 0, 0],
            pointTime = 8.0),
        YumiPosture(
            positionAbsolute = [0.45, 0.0, 0.2],
            orientationAbsolute = np.roll(trans.quaternion_about_axis(np.pi, [np.cos(np.pi/8), np.sin(np.pi/8), 0]), 1),
            pointTime = 8.0),
        YumiPosture(
            positionAbsolute = [0.45, 0.0, 0.2],
            orientationAbsolute = [0, 1, 0, 0],
            pointTime = 8.0),
    ]
    pub.publish(msg)
    
    print("sent msg (coordinated)")
    rospy.sleep(26)


if __name__ == "__main__":
    main()


