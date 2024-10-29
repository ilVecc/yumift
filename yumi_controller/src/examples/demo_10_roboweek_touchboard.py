#!/usr/bin/env python3
import numpy as np

import rospy
import tf.transformations as trans

from yumi_controller.msg import YumiTrajectoryPoint, YumiTrajectory


def main():
    # starting ROS node and subscribers
    rospy.init_node("demo_touchboard", anonymous=True)
    pub = rospy.Publisher("/trajectory", YumiTrajectory, queue_size=1, latch=True)
    rospy.sleep(0.1)

    # --------------------------------------------------

    # TODO this breaks everything
    msg = YumiTrajectory()
    msg.header.stamp = rospy.Time.now()
    msg.mode = "coordinated"
    msg.trajectory = [
        YumiTrajectoryPoint(
            positionAbsolute = [0.45, 0.0, 0.3],
            orientationAbsolute = [1, 0, 0, 0],
            positionRelative = [0, 0.22, 0],
            orientationRelative = trans.quaternion_about_axis(np.radians(170), [1, 0, 0]),
            pointTime = 8.0),
    ]
    pub.publish(msg)
    
    # msg = YumiTrajectory()
    # msg.header.stamp = rospy.Time.now()
    # msg.mode = "individual"
    # msg.trajectory = [
    #     YumiTrajectoryPoint(
    #         positionRight = [0.45, -0.11, 0.3],
    #         orientationRight = trans.quaternion_about_axis(np.radians(-90), [1, 0, 0]),
    #         positionLeft  = [0.45, 0.11, 0.3],
    #         orientationLeft = trans.quaternion_about_axis(np.radians(90), [1, 0, 0]),
    #         gripperRight = 20.,
    #         gripperLeft  = 20.,
    #         pointTime = 10.0),
    #     YumiTrajectoryPoint(
    #         positionRight = [0.45, -0.11, 0.3],
    #         orientationRight = trans.quaternion_about_axis(np.radians(-90), [1, 0, 0]),
    #         positionLeft  = [0.45, 0.11, 0.3],
    #         orientationLeft = trans.quaternion_about_axis(np.radians(90), [1, 0, 0]),
    #         gripperRight = 0.,
    #         gripperLeft  = 0.,
    #         pointTime = 2.0),
    # ]
    # pub.publish(msg)
    
    print("positioning message sent")
    rospy.sleep(10)


if __name__ == "__main__":
    main()


