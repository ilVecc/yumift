#!/usr/bin/env python3
import rospy

from controller.msg import Trajectory_point, Trajectory_msg


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
            positionRight = [0.35, -0.1, 0.04],
            positionLeft = [0.35, 0.1, 0.04],
            orientationRight = [1, 0, 0, 0],
            orientationLeft = [1, 0, 0, 0],
            gripperRight = 0,
            gripperLeft = 0,
            pointTime = 6.0),
        Trajectory_point(
            positionRight = [0.35, -0.15, 0.04],
            positionLeft = [0.35, 0.15, 0.04],
            orientationRight = [1, 0, 0, 0],
            orientationLeft = [1, 0, 0, 0],
            gripperRight = 20,
            gripperLeft = 20,
            pointTime = 8.0),
        Trajectory_point(
            positionRight = [0.35, -0.15, 0.04],
            positionLeft = [0.35, 0.15, 0.04],
            orientationRight = [1, 0, 0, 0],
            orientationLeft = [1, 0, 0, 0],
            gripperRight = 0,
            gripperLeft = 0,
            pointTime = 2.0),
    ]
    pub.publish(msg)

    print("sent msg 1 (individual)")
    rospy.sleep(31)

    # --------------------------------------------------

    msg = Trajectory_msg()
    msg.header.stamp = rospy.Time.now()
    msg.mode = "coordinated"
    msg.trajectory = [
        Trajectory_point(
            positionAbsolute = [0.35, 0.0, 0.2],
            positionRelative = [0, 0.30, 0],
            orientationAbsolute = [1, 0, 0, 0],
            orientationRelative = [0, 0, 0, 1],
            pointTime = 6.0),
        Trajectory_point(
            positionAbsolute = [0.45, 0.1, 0.2],
            positionRelative = [0, 0.30, 0],
            orientationAbsolute = [1, 0, 0, 0],
            orientationRelative = [0, 0, 0, 1],
            pointTime = 6.0),
        Trajectory_point(
            positionAbsolute = [0.45, -0.1, 0.4],
            positionRelative = [0, 0.30, 0],
            orientationAbsolute = [1, 0, 0, 0],
            orientationRelative = [0, 0, 0, 1],
            pointTime = 6.0),
        Trajectory_point(
            positionAbsolute = [0.35, 0.0, 0.2],
            positionRelative = [0, 0.30, 0],
            orientationAbsolute = [1, 0, 0, 0],
            orientationRelative = [0, 0, 0, 1],
            pointTime = 6.0)
    ]
    pub.publish(msg)
    
    print("sent msg 2 (coordinated)")
    rospy.sleep(33)

    # --------------------------------------------------

    msg = Trajectory_msg()
    msg.header.stamp = rospy.Time.now()
    msg.mode = "individual"
    msg.trajectory = [
        Trajectory_point(
            positionRight = [0.35, -0.2, 0.2],
            positionLeft = [0.35, 0.2, 0.2],
            orientationRight = [1, 0, 0, 0],
            orientationLeft = [1, 0, 0, 0],
            gripperRight = 20.0,
            gripperLeft = 20.0,
            pointTime = 8.0)
    ]
    pub.publish(msg)

    print("sent msg 3 (individual)")
    rospy.sleep(31)


if __name__ == "__main__":
    main()


