#!/usr/bin/env python3
import numpy as np

import rospy
import tf.transformations as trans

from yumi_controller.msg import Trajectory_point, Trajectory_msg


def main():
    # starting ROS node and subscribers
    rospy.init_node("trajectory_test", anonymous=True)
    pub = rospy.Publisher("/trajectory", Trajectory_msg, queue_size=1)
    rospy.sleep(0.1)

    # --------------------------------------------------

    msg = Trajectory_msg()                       # message will contain list of trajectory points
    msg.header.stamp = rospy.Time.now()
    msg.mode = "individual"                      # control mode
    msg.trajectory = [
        Trajectory_point(
            positionRight = [0.35, -0.2, 0.2],   # position right arm [m], yumi_base_link is the origin
            positionLeft  = [0.35,  0.2, 0.2],   # position left arm [m]
            orientationRight = [1, 0, 0, 0],     # orientation right arm, quaterniorns [x, y, z, w]
            orientationLeft  = [1, 0, 0, 0],     # orientation left arm
            gripperRight = 0.0,                  # gripper width for the fingers [mm]
            gripperLeft  = 0.0,
            pointTime = 5.0),                    # time to get to this point [s]
        Trajectory_point(
            positionRight = [0.35, -0.1, 0.04],
            positionLeft  = [0.35,  0.1, 0.04],
            orientationRight = [1, 0, 0, 0],
            orientationLeft  = [1, 0, 0, 0],
            pointTime = 2.0),
        Trajectory_point(
            positionRight = [0.35, -0.10, 0.04],
            positionLeft  = [0.35,  0.15, 0.04],
            orientationRight = [1, 0, 0, 0],
            orientationLeft  = [1, 0, 0, 0],
            gripperRight = 20,
            gripperLeft  = 20,
            pointTime = 2.0),
        Trajectory_point(
            positionRight = [0.35, -0.10, 0.04],
            positionLeft  = [0.35,  0.15, 0.04],
            orientationRight = [1, 0, 0, 0],
            orientationLeft  = [1, 0, 0, 0],
            pointTime = 2.0)
    ]

    pub.publish(msg)
    print("sent msg 1, individual")
    rospy.sleep(15)

    # --------------------------------------------

    msg = Trajectory_msg()
    msg.header.stamp = rospy.Time.now()
    msg.mode = "coordinated"  # now controlling with the coordinated motion mode
    msg.trajectory = [
        Trajectory_point(
            positionAbsolute = [0.35, 0.0, 0.2],  # absolute  position, the avg of the gripper positions
            positionRelative = [0, 0.25, 0],      # relatibe  position, the difference of the gripper positions in the absolute frame
            orientationAbsolute = [1, 0, 0, 0],
            orientationRelative = [0, 0, 0, 1],
            pointTime = 5.0),
        Trajectory_point(
            positionAbsolute = [0.45, 0.1, 0.2],
            positionRelative = [0, 0.25, 0],
            orientationAbsolute = [1, 0, 0, 0],
            orientationRelative = [0, 0, 0, 1],
            pointTime = 5.0),
        Trajectory_point(
            positionAbsolute = [0.45, 0.1, 0.4],
            positionRelative = [0, 0.25, 0],
            orientationAbsolute = [1, 0, 0, 0],
            orientationRelative = [0, 0, 0, 1],
            pointTime = 5.0),
        Trajectory_point(
            positionAbsolute = [0.35, 0.0, 0.2],
            positionRelative = [0, 0.25, 0],
            orientationAbsolute = [1, 0, 0, 0],
            orientationRelative = [0, 0, 0, 1],
            pointTime = 5.0)
    ]

    pub.publish(msg)
    print("sent msg 2, coordinated ")
    rospy.sleep(25)

    # --------------------------------------------

    msg = Trajectory_msg()
    msg.header.stamp = rospy.Time.now()
    msg.mode = "coordinated"
    msg.trajectory = [
        Trajectory_point(
            positionAbsolute = [0.4, 0.0, 0.2],
            positionRelative = [0, 0.25, 0],
            orientationAbsolute = trans.quaternion_from_euler(0, 0, np.radians(210), "rzyx"),
            orientationRelative = [0, 0, 0, 1],
            pointTime = 8.0),
        Trajectory_point(
            positionAbsolute = [0.4, 0.0, 0.2],
            positionRelative = [0, 0.25, 0],
            orientationAbsolute = trans.quaternion_from_euler(0, 0, np.radians(150), "rzyx"),
            orientationRelative = [0, 0, 0, 1],
            pointTime = 8.0),
        Trajectory_point(
            positionAbsolute = [0.4, 0.0, 0.2],
            positionRelative = [0, 0.25, 0],
            orientationAbsolute = trans.quaternion_from_euler(0, 0, np.radians(180), "rzyx"),
            orientationRelative = [0, 0, 0, 1],
            pointTime = 8.0),
        Trajectory_point(
            positionAbsolute = [0.4, 0.0, 0.2],
            positionRelative = [0, 0.25, 0],
            orientationAbsolute = trans.quaternion_from_euler(np.radians(40), 0, np.radians(180), "rzyx"),
            orientationRelative = [0, 0, 0, 1],
            pointTime = 8.0),
        Trajectory_point(
            positionAbsolute = [0.4, 0.0, 0.2],
            positionRelative = [0, 0.25, 0],
            orientationAbsolute = trans.quaternion_from_euler(np.radians(-40), 0, np.radians(180), "rzyx"),
            orientationRelative = [0, 0, 0, 1],
            pointTime = 8.0),
        Trajectory_point(
            positionAbsolute = [0.4, 0.0, 0.2],
            positionRelative = [0, 0.25, 0],
            orientationAbsolute = trans.quaternion_from_euler(0, 0, np.radians(180), "rzyx"),
            orientationRelative = [0, 0, 0, 1],
            pointTime = 8.0),
        Trajectory_point(
            positionAbsolute = [0.4, 0.0, 0.2],
            positionRelative = [0, 0.25, 0],
            orientationAbsolute = trans.quaternion_from_euler(0, np.radians(40), np.radians(180), "rzyx"),
            orientationRelative = [0, 0, 0, 1],
            pointTime = 8.0),
        Trajectory_point(
            positionAbsolute = [0.4, 0.0, 0.2],
            positionRelative = [0, 0.25, 0],
            orientationAbsolute = trans.quaternion_from_euler(0, np.radians(-40), np.radians(180), "rzyx"),
            orientationRelative = [0, 0, 0, 1],
            pointTime = 16.0),
        Trajectory_point(
            positionAbsolute = [0.4, 0.0, 0.2],
            positionRelative = [0, 0.25, 0],
            orientationAbsolute = trans.quaternion_from_euler(0, 0, np.radians(180), "rzyx"),
            orientationRelative = [0, 0, 0, 1],
            pointTime = 8.0),
        Trajectory_point(
            positionAbsolute = [0.4, 0.0, 0.2],
            positionRelative = [0, 0.25, 0],
            orientationAbsolute = trans.quaternion_from_euler(0, 0, np.radians(180), "rzyx"),
            orientationRelative = trans.quaternion_from_euler(0, 0, np.radians(60), "rzyx"),
            pointTime = 8.0),
        Trajectory_point(
            positionAbsolute = [0.4, 0.0, 0.2],
            positionRelative = [0, 0.25, 0],
            orientationAbsolute = trans.quaternion_from_euler(0, 0, np.radians(180), "rzyx"),
            orientationRelative = [0, 0, 0, 1],
            pointTime = 8.0)
    ]

    pub.publish(msg)
    print("sent msg 3, coordinated ")
    rospy.sleep(96)
    
    # --------------------------------------------

    msg = Trajectory_msg()
    msg.header.stamp = rospy.Time.now()
    msg.mode = "routine_reset_pose"
    
    pub.publish(msg)
    print("resetPose")


if __name__ == "__main__":
    main()
