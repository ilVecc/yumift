#!/usr/bin/env python3
import numpy as np

import rospy
import tf.transformations as trans

from yumi_controller.msg import YumiPosture, YumiTrajectory


def eul2quat(ai, aj, ak, axes="rzyx"):
    return np.roll(trans.quaternion_from_euler(np.radians(ai), np.radians(aj), np.radians(ak), axes), 1)

def main():
    # starting ROS node and subscribers
    rospy.init_node("trajectory_test", anonymous=True)
    pub = rospy.Publisher("/trajectory", YumiTrajectory, queue_size=1, latch=True)
    rospy.sleep(0.1)

    # --------------------------------------------------

    msg = YumiTrajectory()                       # message will contain list of trajectory points
    msg.header.stamp = rospy.Time.now()
    msg.mode = "individual"                      # control mode
    msg.trajectory = [
        YumiPosture(
            positionRight = [0.35, -0.2, 0.2],   # position right arm [m], yumi_base_link is the origin
            positionLeft  = [0.35,  0.2, 0.2],   # position left arm [m]
            orientationRight = [0, 1, 0, 0],     # orientation right arm, quaterniorns [w, x, y, z]
            orientationLeft  = [0, 1, 0, 0],     # orientation left arm
            gripperRight = 0.0,                  # gripper width for the fingers [mm]
            gripperLeft  = 0.0,
            pointTime = 5.0),                    # time to get to this point [s]
        YumiPosture(
            positionRight = [0.35, -0.1, 0.04],
            positionLeft  = [0.35,  0.1, 0.04],
            orientationRight = eul2quat(0, 45, -135, "rzyx"),
            orientationLeft  = eul2quat(0, 45, 135, "rzyx"),
            pointTime = 5.0),
        YumiPosture(
            positionRight = [0.45, -0.15, 0.15],
            positionLeft  = [0.45,  0.15, 0.15],
            orientationRight = [0, 0, 1, 0],
            orientationLeft  = [0, 0, 1, 0],
            pointTime = 5.0)
    ]

    pub.publish(msg)
    print("sent msg")
    rospy.sleep(15)

    # --------------------------------------------


if __name__ == "__main__":
    main()
