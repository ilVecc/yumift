#!/usr/bin/env python3
import rospy

from yumift_msgs.helper import Helper as H


if __name__ == "__main__":
    rospy.init_node("example_3_reference_frames_advanced", anonymous=True)
    
    # Add a frame 
    H.add_frame("table", H.cm(30, 0, 10), H.e2q(a1=180), H.FRAME_DEFAULT)
    
    H.quick_send(
        topic="/trajectory",
        print_message="Reference frames and local coordinates",
        motion_override=H.Motion.INDIVIDUAL,
        postures=[
            H.posture( 5.0,
                (H.cm(y=+15, z=10), H.e2q()),
                (H.cm(y=-25, z=5), H.e2q()),
                reference=H.Reference.LOCAL,
                frame="table"),
            H.pause(1.0),
            H.posture( 5.0,
                H.e2q(a1=+30),
                H.e2q(a1=-30),
                reference=H.Reference.TOOL),
            H.pause(1.0),
            H.posture( 5.0,
                (H.cm(y=+10, z=-10), H.e2q(a2=-15)),
                (H.cm(y=+10, z=-10), H.e2q(a2=-15)),
                reference=H.Reference.FRAME,
                frame="table"),
            H.pause(1.0),
            H.posture( 5.0,
                (H.cm(x=+10, z=-20), H.e2q(a2=-30)),
                (H.cm(x=+10, z=-20), H.e2q(a2=-30)),
                reference=H.Reference.GLOBAL),
        ])
    