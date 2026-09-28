#!/usr/bin/env python3
import rospy

from yumift_msgs.helper import Helper as H


if __name__ == "__main__":
    rospy.init_node("trajectory_test", anonymous=True)
    
    H.quick_send(
        topic="/trajectory",
        print_message="Exploiting relative nullspace!",
        # Here we set only the absolute frame as a task, informing the controller
        # that the relative frame DOFs can be used as nullspace in the inverse 
        # kinemtics solver algorithm. This means that the controller can actively 
        # choose to set the relative frame as desired to maximize another function
        # (e.g. absolute manipulability), or to simply ignore controlling it.
        # Not all controllers necessarily implement this "nullspace-aware" control
        # law, so be sure to use a compatible controller.
        mode=H.Mode.ABSOLUTE,
        postures=[
            H.posture( 5.0,
                (H.cm(x=45, z=20), H.e2q(ai=180)),
                None,
                H.Mode.ABSOLUTE),
            H.posture( 5.0,
                H.e2q(ai=-45),
                None,
                H.Mode.ABSOLUTE,
                H.Incr.LOCAL),
            H.posture( 5.0,
                (H.cm(x=45, z=20), H.e2q(ai=180)),
                None,
                H.Mode.ABSOLUTE),
        ])
