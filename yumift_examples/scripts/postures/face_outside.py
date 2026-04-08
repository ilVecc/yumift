#!/usr/bin/env python3
import rospy

from yumift_msgs.helper import Helper as H


if __name__ == "__main__":
    rospy.init_node("routine_face_outside", anonymous=True)

    H.quick_send(
        topic="/trajectory",
        print_message="FACE_OUT sent",
        postures=[
            H.posture( 4.0,
            ([0.45, -0.35, 0.5], H.e2q(+90, 0, 0, "sxyz")),
            ([0.45, +0.35, 0.5], H.e2q(-90, 0, 0, "sxyz")))
    ])
