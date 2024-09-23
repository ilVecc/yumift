#!/usr/bin/env python3
import rospy

from yumi_controller.msg import Trajectory_point, Trajectory_msg


def main():
    # starting ROS node and subscribers
    rospy.init_node("trajectory_test", anonymous=True)
    pub = rospy.Publisher("/trajectory", Trajectory_msg, queue_size=1)
    rospy.sleep(0.1)

    # --------------------------------------------------

    msg = Trajectory_msg()
    msg.header.stamp = rospy.Time.now()
    msg.mode = "coordinated"
    msg.trajectory = [
        Trajectory_point(
            positionAbsolute = [0.35, 0, 0.20],
            positionRelative = [0, 0.25, 0],
            orientationAbsolute = [1, 0, 0, 0],
            orientationRelative = [0, 0, 0, 1],
            pointTime = 8.0),
        Trajectory_point(
            local = True,                                 # this point is defined w.r.t. to the previous one
            positionAbsolute = [0.10, 0, 0],              # [0.35, 0.0, 0.20] + [0.10, 0.0, 0.0] = [0.45, 0.0, 0.20]
            pointTime = 8.0),
        Trajectory_point(
            local = True,
            positionAbsolute = [0, 0.10, 0.20],           # [0.45, 0.0, 0.20] + [0, 0.10, 0.20] = [0.45, 0.10, 0.40]
            orientationAbsolute = [1, 0, 0, 0],           # [1, 0, 0, 0] * [1, 0, 0, 0] = [0, 0, 0, 1]
            pointTime = 8.0),
        Trajectory_point(
            local = True,
            positionAbsolute = [-0.10, -0.10, -0.20],
            orientationAbsolute = [1, 0, 0, 0],           # [1, 0, 0, 0] * [0, 0, 0, 1] = [1, 0, 0, 0]
            pointTime = 8.0),
    ]

    pub.publish(msg)
    print("Sent message (coordinated)")
    rospy.sleep(32)

    # --------------------------------------------------

    msg = Trajectory_msg()
    msg.header.stamp = rospy.Time.now()
    msg.mode = "routine_reset_pose"

    pub.publish(msg)
    print("Sent reset_pose command")


if __name__ == "__main__":
    main()
