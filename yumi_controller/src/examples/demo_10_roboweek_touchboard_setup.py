#!/usr/bin/env python3
import numpy as np

import rospy
import tf.transformations as trans

from yumi_controller.msg import YumiPosture, YumiTrajectory


def main():
    # starting ROS node and subscribers
    rospy.init_node("demo_touchboard_setup", anonymous=True)
    pub = rospy.Publisher("/trajectory", YumiTrajectory, queue_size=1, latch=True)
    rospy.sleep(0.1)

    # --------------------------------------------------

    msg = YumiTrajectory()
    msg.mode = "individual"
    msg.header.stamp = rospy.Time.now()
    msg.trajectory = [
        YumiPosture(
            positionRight = [0.45, -0.10, 0.3],
            orientationRight = np.roll(trans.quaternion_about_axis(np.radians(-90), [1, 0, 0]), 1),
            positionLeft  = [0.45, 0.10, 0.3],
            orientationLeft = np.roll(trans.quaternion_about_axis(np.radians(90), [1, 0, 0]), 1),
            gripperRight = 20.,
            gripperLeft  = 20.,
            pointTime = 5.0),
    ]
    pub.publish(msg)
    print("positioning message sent")
    rospy.sleep(5)

    # # keep holding the wood board
    # while not rospy.is_shutdown():
    #     msg.header.stamp = rospy.Time.now()
    #     msg.trajectory = [
    #         YumiPosture(
    #             positionRight = [0.45, -0.10, 0.3],
    #             orientationRight = np.roll(trans.quaternion_about_axis(np.radians(-90), [1, 0, 0]), 1),
    #             positionLeft  = [0.45, 0.10, 0.3],
    #             orientationLeft = np.roll(trans.quaternion_about_axis(np.radians(90), [1, 0, 0]), 1),
    #             gripperRight = 0.,
    #             gripperLeft  = 0.,
    #             pointTime = 1.0),
    #     ]
    #     pub.publish(msg)
    #     print("holding message sent")
    #     rospy.sleep(5 * 60)


if __name__ == "__main__":
    main()


