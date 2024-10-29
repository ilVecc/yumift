#!/usr/bin/env python3
import rospy

from yumi_controller.msg import YumiTrajectoryPoint, YumiTrajectory


def main():
    # starting ROS node and subscribers
    rospy.init_node("routine_move_down", anonymous=True)
    pub = rospy.Publisher("/trajectory", YumiTrajectory, queue_size=1)
    rospy.sleep(0.1)

    # --------------------------------------------------

    msg = YumiTrajectory()
    msg.header.stamp = rospy.Time.now()
    msg.mode = "coordinated"
    msg.trajectory = [
        YumiTrajectoryPoint(
            positionAbsolute = [0.4, 0.0, 0.0],
            orientationAbsolute = [0, 1, 0, 0],
            positionRelative = [0.0,  0.4, 0.0],
            orientationRelative  = [1, 0, 0, 0],
            pointTime = 4.0),
    ]
    pub.publish(msg)
    rospy.sleep(4)


if __name__ == "__main__":
    main()


