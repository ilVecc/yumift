#!/usr/bin/env python3
import numpy as np

import rospy
import tf.transformations as trans

from yumi_controller.msg import YumiPosture, YumiTrajectory


def main():
    # starting ROS node and subscribers
    rospy.init_node("trajectory_test", anonymous=True)
    pub = rospy.Publisher("/trajectory", YumiTrajectory, queue_size=1, latch=True)
    rospy.sleep(0.1)

    # --------------------------------------------------

    msg = YumiTrajectory()
    msg.header.stamp = rospy.Time.now()
    msg.mode = "coordinated"
    msg.trajectory = [
        # YumiPosture(
        #     positionAbsolute = [0.45, 0.0, 0.2],
        #     positionRelative = [0, 0.30, 0],
            # orientationAbsolute = [0, 1, 0, 0],
            # orientationRelative = [1, 0, 0, 0],
        #     pointTime = 4.0),
        # YumiPosture(
        #     positionAbsolute = [0.55, 0.1, 0.2],
        #     positionRelative = [0, 0.30, 0],
            # orientationAbsolute = [0, 1, 0, 0],
            # orientationRelative = [1, 0, 0, 0],
        #     pointTime = 4.0),
        # YumiPosture(
        #     positionAbsolute = [0.55, -0.1, 0.4],
        #     positionRelative = [0, 0.30, 0],
            # orientationAbsolute = [0, 1, 0, 0],
            # orientationRelative = [1, 0, 0, 0],
        #     pointTime = 4.0),
        YumiPosture(
            positionAbsolute = [0.45, 0.0, 0.2],
            positionRelative = [0, 0.30, 0],
            orientationAbsolute = [0, 1, 0, 0],
            orientationRelative = [1, 0, 0, 0],
            pointTime = 8.0),
        YumiPosture(
            positionAbsolute = [0.45, 0.0, 0.2],
            positionRelative = [0, 0.30, 0],
            orientationAbsolute = np.roll(trans.quaternion_about_axis(np.pi, [np.cos(np.pi/8), np.sin(np.pi/8), 0]), 1),
            orientationRelative = [1, 0, 0, 0],
            pointTime = 8.0),
        YumiPosture(
            positionAbsolute = [0.45, 0.0, 0.2],
            positionRelative = [0, 0.30, 0],
            orientationAbsolute = [0, 1, 0, 0],
            orientationRelative = [1, 0, 0, 0],
            pointTime = 8.0),
    ]
    pub.publish(msg)
    
    print("sent msg (coordinated)")
    rospy.sleep(26)


if __name__ == "__main__":
    main()


