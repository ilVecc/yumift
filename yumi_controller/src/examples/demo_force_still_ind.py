#!/usr/bin/env python
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
    msg.mode = "individual"
    msg.trajectory = [
        Trajectory_point(
            positionRight = [0.5, -0.2, 0.2],
            positionLeft  = [0.5,  0.2, 0.2],
            orientationRight = [1, 0, 0, 0],
            orientationLeft  = [1, 0, 0, 0],
            gripperRight = 20.0,
            gripperLeft  = 20.0,
            pointTime = 8.0),
        Trajectory_point(
            positionRight = [0.5, -0.2, 0.2],
            positionLeft  = [0.5,  0.2, 0.2],
            orientationRight = [1, 0, 0, 0],
            orientationLeft  = [1, 0, 0, 0],
            pointTime = 12.0),
    ]
    pub.publish(msg)

    print("Sent message (individual)")
    rospy.sleep(31)


if __name__ == "__main__":
    main()

