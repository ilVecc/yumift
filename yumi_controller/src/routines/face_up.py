#!/usr/bin/env python3
import rospy

from yumi_controller.msg import YumiTrajectoryPoint, YumiTrajectory


def main():
    # starting ROS node and subscribers
    rospy.init_node("routine_face_down", anonymous=True)
    pub = rospy.Publisher("/trajectory", YumiTrajectory, queue_size=1)
    rospy.sleep(0.1)

    # --------------------------------------------------

    msg = YumiTrajectory()
    msg.header.stamp = rospy.Time.now()
    msg.mode = "individual"
    msg.trajectory = [
        YumiTrajectoryPoint(
            positionRight = [0.4, -0.3, 0.6],
            orientationRight = [0, 0, 0, 1],
            positionLeft  = [0.4,  0.3, 0.6],
            orientationLeft  = [0, 0, 0, 1],
            pointTime = 4.0),
    ]
    pub.publish(msg)
    rospy.sleep(4)


if __name__ == "__main__":
    main()


