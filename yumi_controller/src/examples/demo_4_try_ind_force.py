#!/usr/bin/env python
import rospy

from yumi_controller.msg import YumiPosture, YumiTrajectory


def main():
    # starting ROS node and subscribers
    rospy.init_node("trajectory_test", anonymous=True)
    pub = rospy.Publisher("/trajectory", YumiTrajectory, queue_size=1)
    rospy.sleep(0.1)

    # --------------------------------------------------

    msg = YumiTrajectory()
    msg.header.stamp = rospy.Time.now()
    msg.mode = "individual"
    msg.trajectory = [
        YumiPosture(
            positionRight = [0.5, -0.2, 0.2],
            positionLeft  = [0.5,  0.2, 0.2],
            orientationRight = [0, 1, 0, 0],
            orientationLeft  = [0, 1, 0, 0],
            gripperRight = 20.0,
            gripperLeft  = 20.0,
            pointTime = 8.0),
        YumiPosture(
            positionRight = [0.5, -0.2, 0.2],
            positionLeft  = [0.5,  0.2, 0.2],
            orientationRight = [0, 1, 0, 0],
            orientationLeft  = [0, 1, 0, 0],
            pointTime = 12.0),
    ]
    pub.publish(msg)

    print("Sent message (individual)")
    rospy.sleep(31)


if __name__ == "__main__":
    main()


