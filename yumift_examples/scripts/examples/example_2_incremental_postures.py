#!/usr/bin/env python3
import rospy

from yumift_msgs.helper import Helper as H

# This second example is a follow-up to the first one and demostrates how to 
# send incremental trajectories.
# 
# As seen previously, a trajectory is a list of global postures with a duration. 
# Sometime, it's more useful to express the next point along the trajectory as a 
# movement from the previous one, so to avoid manually computing each posture.
# This can be conveniently achieved with the `incremental` keyword.

if __name__ == "__main__":
    # First, initialize the ROS node
    rospy.init_node("example_2_incremental_postures", anonymous=True)
    
    # Then, as seen in the previous example, quickly contruct and send a trajectory
    H.quick_send(
        topic="/trajectory",
        print_message="Incremental trajectory sent",
        wait_completion=True,
        postures=[
            H.posture( 8.0,
                primary=([0.35, -0.2, 0.2], H.e2q(-90, 0, 0)),
                secondary=([0.35, +0.2, 0.2], H.e2q(+90, 0, 0)),
                incremental=H.Incr.OFF),
            # These three points are expressed as an increment to their previous 
            # point, w.r.t. the (global) world frame, i.e. Yumi's base.
            # Essentially, we are now passing a transformation (translation and 
            # rotation) instead of a pose (position and orientation).
            H.posture( 5.0,
                # [0.35, -0.2, 0.2]_WORLD + [0, +0.10, 0]_WORLD
                primary=([0, +0.10, 0], H.e2q(0, 0, -45, "rxyz")),
                # no explicit secondary pose provided, this defaults to "do nothing"
                secondary=tuple(),
                # the increment is expressed in the world frame
                incremental=H.Incr.GLOBAL),
            H.posture( 5.0,
                primary=([0, 0, -0.10], H.e2q(0, 45, 0, "rxyz")),
                incremental=H.Incr.GLOBAL),
            H.posture( 5.0,
                primary=([+0.1, 0, 0], H.e2q(45, 0, 0, "rxyz")),
                incremental=H.Incr.GLOBAL),
            # Handy method for "keep previous posture for X seconds" message
            H.pause(2.0),
            # These three points are expressed as an increment to their previous
            # point, w.r.t. the (local) target frame, i.e. Yumi's left tooltip.
            # The local increments are internally converted to global and then 
            # applied to the previous point's global coordinates. 
            H.posture( 5.0,
                # [x y z]_WORLD + [0, +0.10, 0]_LOCAL
                secondary=([0, +0.10, 0], H.e2q(0, 0, -45, "rxyz")),
                incremental=H.Incr.LOCAL),
            H.posture( 5.0,
                # the rotation is computed along the gripper's y-axis
                secondary=([0, 0, -0.10], H.e2q(0, 45, 0, "rxyz")),
                incremental=H.Incr.LOCAL),
            H.posture( 5.0,
                # a "skrew-like" motion (a true skrew motion cannot be achieved, 
                # as here we are just defining orientations, not trajectories; 
                # the "skew-like" effect is thus simply a cool byproduct)
                secondary=([0, 0, +0.20], H.e2q(0, 0, -180, "rxyz")),
                incremental=H.Incr.LOCAL),
    ])
