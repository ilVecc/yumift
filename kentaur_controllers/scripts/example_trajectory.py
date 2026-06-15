#!/usr/bin/env python3
import rospy

from yumift_msgs.helper import Helper as H

if __name__ == "__main__":
    rospy.init_node("example_whole_body_trajectory", anonymous=True)
    
    H.quick_send(
        topic="/trajectory",
        print_message="sent trajectory",
        postures=[
            H.posture(5.0, (H.cm(-20, +20, +20), H.e2q(ai=0)), (H.cm(-20, -20, +20), H.e2q(ai=0)), incremental=H.Incr.GLOBAL),
            H.posture(5.0, (H.cm(-40, -10, -20), H.e2q(ai=0)), (H.cm(-40, +10, -20), H.e2q(ai=0)), incremental=H.Incr.GLOBAL),
    ])
