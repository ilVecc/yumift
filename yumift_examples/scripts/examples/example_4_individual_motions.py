#!/usr/bin/env python3
import rospy

from yumift_msgs.helper import Helper as H


if __name__ == "__main__":
    rospy.init_node("example_4_individual_motions", anonymous=True)

    H.quick_send(
        topic="/trajectory",
        print_message="Starting from a good position... ",
        motion_default=H.Motion.INDIVIDUAL,
        postures=[
            H.posture( 5.0,
                (H.cm(x=30, y=-20, z=20), H.e2q(a1=180)),
                (H.cm(x=30, y=+20, z=20), H.e2q(a1=180)),
                H.Motion.INDIVIDUAL),
        ])
    
    H.quick_send(
        topic="/trajectory",
        print_message="... move right arm only (left is still subject to secondary objectives)",
        motion_default=H.Motion.RIGHT,
        postures=[
            H.posture( 3.0,
                H.e2q(a1=+90),
                None,
                H.Motion.RIGHT,
                H.Reference.TOOL),
            H.posture( 2.0,
                H.e2q(a1=-45),
                None,
                H.Motion.RIGHT,
                H.Reference.TOOL),
            H.posture( 1.0,
                H.e2q(a1=+15),
                None,
                H.Motion.RIGHT,
                H.Reference.TOOL),
            H.posture( 2.0,
                H.e2q(a1=-90),
                None,
                H.Motion.RIGHT,
                H.Reference.TOOL),
            H.posture( 2.0,
                H.e2q(a1=+75),
                None,
                H.Motion.RIGHT,
                H.Reference.TOOL),
        ])
