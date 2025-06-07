#!/usr/bin/env python3
import rospy

from yumi_controller.msg import YumiPosture, YumiTrajectory


def main():
    # starting ROS node and subscribers
    rospy.init_node("trajectory_test", anonymous=True)
    pub = rospy.Publisher("/trajectory", YumiTrajectory, queue_size=1, latch=True)
    rospy.sleep(0.1)

    # --------------------------------------------------

    msg = YumiTrajectory()
    msg.header.stamp = rospy.Time.now()
    msg.mode = "coordinated"
    msg.trajectory = [
        YumiPosture(
            positionAbsolute = [0.45, 0.0, 0.2],
            positionRelative = [0, 0.30, 0],
            orientationRelative = [1, 0, 0, 0],
            orientationAbsolute = [0, 1, 0, 0],
            gripperRight = 20.0,
            gripperLeft  = 20.0,
            pointTime = 8.0),
        YumiPosture(
            positionAbsolute = [0.45, 0.0, 0.2],
            positionRelative = [0, 0.30, 0],
            orientationAbsolute = [0, 1, 0, 0],
            orientationRelative = [1, 0, 0, 0],
            pointTime = 8.0),
    ]
    pub.publish(msg)

    print("sent msg 1 (coordinated)")
    rospy.sleep(31)


if __name__ == "__main__":
    main()


