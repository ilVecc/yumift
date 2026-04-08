#!/usr/bin/env python3
import rospy

from yumift_msgs.helper import Helper as H


if __name__ == "__main__":
    rospy.init_node("routine_face_back", anonymous=True)
    
    H.quick_send(
        topic="/trajectory",
        print_message="FACE_BACK sent",
        postures=[
            H.posture( 4.0,
            ([0.15, -0.25, 0.6], H.e2q(0, -90, 0, "rxyz")),
            ([0.15, +0.25, 0.6], H.e2q(0, -90, 0, "rxyz")))
    ])
