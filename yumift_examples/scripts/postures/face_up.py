#!/usr/bin/env python3
import rospy

from yumift_msgs.helper import Helper as H


if __name__ == "__main__":
    rospy.init_node("routine_face_up", anonymous=True)

    H.quick_send(
        topic="/trajectory",
        print_message="FACE_UP sent",
        postures=[
            H.posture( 4.0,
            ([0.4, -0.2, 0.4], [1, 0, 0, 0]),
            ([0.4, +0.2, 0.4], [1, 0, 0, 0]))
    ])
