#!/usr/bin/env python3
import rospy

from yumift_msgs.helper import Helper as H

if __name__ == "__main__":
    rospy.init_node("routine_abort", anonymous=True)
    H.quick_send(
        topic="/trajectory", 
        routine_name="abort_trajectory", 
        wait_completion=-0.5,
        print_message="ABORT sent")
