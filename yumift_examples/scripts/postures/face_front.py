#!/usr/bin/env python3
import rospy

from yumift_msgs.helper import Helper as H


if __name__ == "__main__":
    rospy.init_node("routine_face_front", anonymous=True)
    
    H.quick_send(
        topic="/trajectory",
        print_message="FACE_FRONT sent",
        postures=[
            H.posture( 4.0,
            ([0.45, -0.25, 0.0], H.e2q(0, 90, 0, "rxyz")),
            ([0.45, +0.25, 0.0], H.e2q(0, 90, 0, "rxyz")))
    ])
