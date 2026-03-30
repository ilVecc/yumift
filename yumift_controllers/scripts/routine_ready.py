#!/usr/bin/env python3
import rospy

from yumift_msgs.yumi_posture_helper import Helper

if __name__ == "__main__":
    rospy.init_node("routine_ready", anonymous=True)
    Helper.quick_send(
        topic="/trajectory", 
        routine_name="ready_pose", 
        print_message="READY_POSE sent", 
        wait_completion=True)
