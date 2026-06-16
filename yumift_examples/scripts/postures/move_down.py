#!/usr/bin/env python3
import rospy

from yumift_msgs.helper import Helper as H


if __name__ == "__main__":
    rospy.init_node("routine_move_down", anonymous=True)
    
    H.quick_send(
        topic="/trajectory",
        print_message="MOVE_DOWN sent",
        mode=H.Mode.COORDINATED,
        postures=[
            H.posture( 4.0,
                (H.cm(x=40), H.e2q(ai=180)),
                (H.cm(y=40), H.e2q()),
                H.Mode.COORDINATED)
        ])
