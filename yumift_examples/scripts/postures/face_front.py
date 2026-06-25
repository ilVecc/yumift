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
            (H.cm(45, -25, 10), H.e2q(aj=90)),
            (H.cm(45, +25, 10), H.e2q(aj=90)))
    ])
