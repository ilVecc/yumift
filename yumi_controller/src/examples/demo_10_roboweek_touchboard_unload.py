#!/usr/bin/env python3
import numpy as np

import rospy
import tf.transformations as trans

from yumi_controller.msg import YumiTrajectoryPoint, YumiTrajectory


def main():
    # starting ROS node and subscribers
    rospy.init_node("demo_touchboard_unload", anonymous=True)
    pub = rospy.Publisher("/trajectory", YumiTrajectory, queue_size=1, latch=True)
    rospy.sleep(0.1)

    # --------------------------------------------------

    msg = YumiTrajectory()
    msg.mode = "individual"
    msg.header.stamp = rospy.Time.now()
    msg.trajectory = [
        YumiTrajectoryPoint(
            positionRight = [-0.2, 0.0, -0.2],
            orientationRight = np.roll(trans.quaternion_about_axis(np.radians(-45), [0, 1, 0]), 1),
            positionLeft  = [-0.2, 0.0, -0.2],
            orientationLeft  = np.roll(trans.quaternion_about_axis(np.radians(-45), [0, 1, 0]), 1),
            gripperRight = 0.,
            gripperLeft  = 0.,
            pointTime = 3.0,
            local = True),
    ]
    pub.publish(msg)
    print("positioning message sent")
    rospy.sleep(5)


if __name__ == "__main__":
    main()


