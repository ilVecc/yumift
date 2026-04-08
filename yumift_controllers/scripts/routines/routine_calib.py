#!/usr/bin/env python3
import rospy

from yumift_msgs.helper import Helper as H

if __name__ == "__main__":
    rospy.init_node("routine_calib", anonymous=True)
    H.quick_send(
        topic="/trajectory", 
        routine_name="calib_pose", 
        print_message="CALIB_POSE sent")
