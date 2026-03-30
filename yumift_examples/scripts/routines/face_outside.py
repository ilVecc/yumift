#!/usr/bin/env python3
import rospy

from yumift_msgs.msg import YumiTrajectory
from yumift_msgs.yumi_posture_helper import Helper


def main():
    # starting ROS node and subscribers
    rospy.init_node("routine_face_outside", anonymous=True)
    pub = rospy.Publisher("/trajectory", YumiTrajectory, queue_size=1, latch=True)
    rospy.sleep(0.1)

    pub.publish(Helper.trajectory(postures=[
        Helper.posture(4.0,
            ([0.45, -0.35, 0.5], Helper.eul2quat(+90, 0, 0, "sxyz")),
            ([0.45, +0.35, 0.5], Helper.eul2quat(-90, 0, 0, "sxyz")))]))
    print("FACE_OUT sent")
    rospy.sleep(4)


if __name__ == "__main__":
    main()
