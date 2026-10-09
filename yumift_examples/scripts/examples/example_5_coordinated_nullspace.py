#!/usr/bin/env python3
import rospy

from yumift_msgs.helper import Helper as H


if __name__ == "__main__":
    rospy.init_node("example_5_coordinated_nullspace", anonymous=True)
    
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
        motion_default=H.Motion.ABSOLUTE,
        postures=[
            H.posture( 5.0,
                (H.cm(x=45, z=20), H.e2q(a1=180)),
                None,
                H.Motion.ABSOLUTE),
            H.posture( 5.0,
                H.e2q(a1=-45),
                None,
                H.Motion.ABSOLUTE,
                H.Reference.TOOL),
            H.posture( 5.0,
                (H.cm(x=45, z=20), H.e2q(a1=180)),
                None,
                H.Motion.ABSOLUTE),
        ])
