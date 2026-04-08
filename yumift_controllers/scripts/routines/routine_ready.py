#!/usr/bin/env python3
import rospy

from yumift_msgs.helper import Helper as H

if __name__ == "__main__":
    rospy.init_node("routine_ready", anonymous=True)
    H.quick_send(
        topic="/trajectory", 
        routine_name="ready_pose", 
        print_message="READY_POSE sent")
