#!/usr/bin/env python3
import numpy as np

import rospy
import tf.transformations as trans

from yumi_controller.msg import YumiPosture, YumiTrajectory


def eul_to_quat(ai, aj, ak, axes):
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
            orientationRight = [0, 1, 0, 0],
            orientationLeft  = [0, 1, 0, 0],
            pointTime = 2.0),
        YumiPosture(
            positionRight = [0.35, -0.10, 0.04],
            positionLeft  = [0.35,  0.15, 0.04],
            orientationRight = [0, 1, 0, 0],
            orientationLeft  = [0, 1, 0, 0],
            gripperRight = 20,
            gripperLeft  = 20,
            pointTime = 2.0),
        YumiPosture(
            positionRight = [0.35, -0.10, 0.04],
            positionLeft  = [0.35,  0.15, 0.04],
            orientationRight = [0, 1, 0, 0],
            orientationLeft  = [0, 1, 0, 0],
            pointTime = 2.0)
    ]

    pub.publish(msg)
    print("sent msg 1, individual")
    rospy.sleep(15)

    # --------------------------------------------

    msg = YumiTrajectory()
    msg.header.stamp = rospy.Time.now()
    msg.mode = "coordinated"  # now controlling with the coordinated motion mode
    msg.trajectory = [
        YumiPosture(
            positionAbsolute = [0.35, 0.0, 0.2],  # absolute  position, the avg of the gripper positions
            positionRelative = [0, 0.25, 0],      # relatibe  position, the difference of the gripper positions in the absolute frame
            orientationAbsolute = [0, 1, 0, 0],
            orientationRelative = [1, 0, 0, 0],
            pointTime = 5.0),
        YumiPosture(
            positionAbsolute = [0.45, 0.1, 0.2],
            positionRelative = [0, 0.25, 0],
            orientationAbsolute = [0, 1, 0, 0],
            orientationRelative = [1, 0, 0, 0],
            pointTime = 5.0),
        YumiPosture(
            positionAbsolute = [0.45, 0.1, 0.4],
            positionRelative = [0, 0.25, 0],
            orientationAbsolute = [0, 1, 0, 0],
            orientationRelative = [1, 0, 0, 0],
            pointTime = 5.0),
        YumiPosture(
            positionAbsolute = [0.35, 0.0, 0.2],
            positionRelative = [0, 0.25, 0],
            orientationAbsolute = [0, 1, 0, 0],
            orientationRelative = [1, 0, 0, 0],
            pointTime = 5.0)
    ]

    pub.publish(msg)
    print("sent msg 2, coordinated")
    rospy.sleep(25)

    # --------------------------------------------

    msg = YumiTrajectory()
    msg.header.stamp = rospy.Time.now()
    msg.mode = "coordinated"
    msg.trajectory = [
        YumiPosture(
            positionAbsolute = [0.4, 0.0, 0.2],
            positionRelative = [0, 0.25, 0],
            orientationAbsolute = eul_to_quat(0, 0, 210, "rzyx"),
            orientationRelative = [1, 0, 0, 0],
            pointTime = 8.0),
        YumiPosture(
            positionAbsolute = [0.4, 0.0, 0.2],
            positionRelative = [0, 0.25, 0],
            orientationAbsolute = eul_to_quat(0, 0, 150, "rzyx"),
            orientationRelative = [1, 0, 0, 0],
            pointTime = 8.0),
        YumiPosture(
            positionAbsolute = [0.4, 0.0, 0.2],
            positionRelative = [0, 0.25, 0],
            orientationAbsolute = eul_to_quat(0, 0, 180, "rzyx"),
            orientationRelative = [1, 0, 0, 0],
            pointTime = 8.0),
        YumiPosture(
            positionAbsolute = [0.4, 0.0, 0.2],
            positionRelative = [0, 0.25, 0],
            orientationAbsolute = eul_to_quat(40, 0, 180, "rzyx"),
            orientationRelative = [1, 0, 0, 0],
            pointTime = 8.0),
        YumiPosture(
            positionAbsolute = [0.4, 0.0, 0.2],
            positionRelative = [0, 0.25, 0],
            orientationAbsolute = eul_to_quat(-40, 0, 180, "rzyx"),
            orientationRelative = [1, 0, 0, 0],
            pointTime = 8.0),
        YumiPosture(
            positionAbsolute = [0.4, 0.0, 0.2],
            positionRelative = [0, 0.25, 0],
            orientationAbsolute = eul_to_quat(0, 0, 180, "rzyx"),
            orientationRelative = [1, 0, 0, 0],
            pointTime = 8.0),
        YumiPosture(
            positionAbsolute = [0.4, 0.0, 0.2],
            positionRelative = [0, 0.25, 0],
            orientationAbsolute = eul_to_quat(0, 40, 180, "rzyx"),
            orientationRelative = [1, 0, 0, 0],
            pointTime = 8.0),
        YumiPosture(
            positionAbsolute = [0.4, 0.0, 0.2],
            positionRelative = [0, 0.25, 0],
            orientationAbsolute = eul_to_quat(0, -40, 180, "rzyx"),
            orientationRelative = [1, 0, 0, 0],
            pointTime = 16.0),
        YumiPosture(
            positionAbsolute = [0.4, 0.0, 0.2],
            positionRelative = [0, 0.25, 0],
            orientationAbsolute = eul_to_quat(0, 0, 180, "rzyx"),
            orientationRelative = [1, 0, 0, 0],
            pointTime = 8.0),
        YumiPosture(
            positionAbsolute = [0.4, 0.0, 0.2],
            positionRelative = [0, 0.25, 0],
            orientationAbsolute = eul_to_quat(0, 0, 180, "rzyx"),
            orientationRelative = eul_to_quat(0, 0, 60, "rzyx"),
            pointTime = 8.0),
        YumiPosture(
            positionAbsolute = [0.4, 0.0, 0.2],
            positionRelative = [0, 0.25, 0],
            orientationAbsolute = eul_to_quat(0, 0, 180, "rzyx"),
            orientationRelative = [1, 0, 0, 0],
            gripperRight = 20,
            gripperLeft  = 20,
            pointTime = 8.0),
        YumiPosture(
            positionAbsolute = [0.4, 0.0, 0.2],
            positionRelative = [0, 0.25, 0],
            orientationAbsolute = eul_to_quat(0, 0, 180, "rzyx"),
            orientationRelative = [1, 0, 0, 0],
            pointTime = 1.0)
    ]

    pub.publish(msg)
    print("sent msg 3, coordinated")
    rospy.sleep(96)
    
    # --------------------------------------------

    msg = YumiTrajectory()
    msg.header.stamp = rospy.Time.now()
    msg.mode = "routine_ready_pose"
    
    pub.publish(msg)
    print("sent \"ready_pose\" routine")


if __name__ == "__main__":
    main()
