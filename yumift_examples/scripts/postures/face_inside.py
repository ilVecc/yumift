#!/usr/bin/env python3
import rospy

from yumift_msgs.helper import Helper as H


if __name__ == "__main__":
    rospy.init_node("routine_face_inside", anonymous=True)
    
    H.quick_send(
        topic="/trajectory",
        print_message="FACE_IN sent",
        postures=[
            H.posture( 4.0,
            ([0.45, -0.15, 0.25], H.e2q(-90, 0, 0, "sxyz")),
            ([0.45, +0.15, 0.25], H.e2q(+90, 0, 0, "sxyz")))
    ])
