#!/usr/bin/env python3
import rospy

from yumift_msgs.helper import Helper as H

if __name__ == "__main__":
    rospy.init_node("example_wiggle", anonymous=True)
    
    angle = 45
    time = 3.0
    pause = 1.0
    
    H.quick_send(
        topic="/trajectory",
        print_message="Wiggle wiggle wiggle",
        wait_completion=True,   # sleep for the trajectory total time
        postures=[
            H.posture( 4.0,
                (H.cm(45, -15, 25), H.e2q(-90, 0, 0, "sxyz")),
                (H.cm(45, +15, 25), H.e2q(+90, 0, 0, "sxyz")),
                reference=H.Reference.BASE),
            H.pause(pause),
            # wiggle x
            H.posture( time,
                (H.cm(), H.e2q(a1=+angle)),
                (H.cm(), H.e2q(a1=-angle)),
                reference=H.Reference.TOOL),
            H.pause(pause),
            H.posture( time,
                (H.cm(), H.e2q(a1=-angle*2)),
                (H.cm(), H.e2q(a1=+angle*2)),
                reference=H.Reference.TOOL),
            H.pause(pause),
            H.posture( time,
                (H.cm(), H.e2q(a1=+angle)),
                (H.cm(), H.e2q(a1=-angle)),
                reference=H.Reference.TOOL),
            H.pause(pause),
            # wiggle y
            H.posture( time,
                (H.cm(), H.e2q(a2=+angle)),
                (H.cm(), H.e2q(a2=+angle)),
                reference=H.Reference.TOOL),
            H.pause(pause),
            H.posture( time,
                (H.cm(), H.e2q(a2=-angle*2)),
                (H.cm(), H.e2q(a2=-angle*2)),
                reference=H.Reference.TOOL),
            H.pause(pause),
            H.posture( time,
                (H.cm(), H.e2q(a2=+angle)),
                (H.cm(), H.e2q(a2=+angle)),
                reference=H.Reference.TOOL),
            H.pause(pause),
            # wiggle z
            H.posture( time,
                (H.cm(), H.e2q(a3=+angle)),
                (H.cm(), H.e2q(a3=+angle)),
                reference=H.Reference.TOOL),
            H.pause(pause),
            H.posture( time,
                (H.cm(), H.e2q(a3=-angle*2)),
                (H.cm(), H.e2q(a3=-angle*2)),
                reference=H.Reference.TOOL),
            H.pause(pause),
            H.posture( time,
                (H.cm(), H.e2q(a3=+angle)),
                (H.cm(), H.e2q(a3=+angle)),
                reference=H.Reference.TOOL),
            H.pause(pause),
    ])
