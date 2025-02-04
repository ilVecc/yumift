#!/usr/bin/env python3
import rospy

from yumi_controller.msg import YumiTrajectory


def main():
    # starting ROS node and subscribers
    rospy.init_node("routine_ready", anonymous=True)
    pub = rospy.Publisher("/trajectory", YumiTrajectory, queue_size=1, latch=True)
    rospy.sleep(0.1)

    # --------------------------------------------------

    msg = YumiTrajectory()
    msg.header.stamp = rospy.Time.now()
    msg.mode = "routine_ready_pose"
    pub.publish(msg)
    print("message sent")
    rospy.sleep(1)


if __name__ == "__main__":
    main()


