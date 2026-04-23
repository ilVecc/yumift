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
                primary=(H.cm(35, -20, 20), H.e2q(-90, 0, 0)),
                secondary=(H.cm(35, +20, 20), H.e2q(+90, 0, 0)),
                incremental=H.Incr.OFF),
            # These three points are expressed as an increment to their previous 
            # point, w.r.t. the (global) world frame, i.e. Yumi's base.
            # Essentially, we are now passing a transformation (translation and 
            # rotation) instead of a pose (position and orientation).
            H.posture( 5.0,
                # [0.35, -0.2, 0.2]_WORLD + [0, +0.10, 0]_WORLD
                primary=(H.cm(0, +10, 0), H.e2q(0, 0, -45, "rxyz")),
                # No explicit secondary pose provided, this defaults to "do nothing"
                # and can also be omitted entirely.
                secondary=tuple(),
                # The increment is expressed in the world frame
                incremental=H.Incr.GLOBAL),
            H.posture( 5.0,
                # To simplify even further the commands, you can avoid writing 
                # the coordinates you do not care about, and default them to 0.
                # For orientations, the default behaviour is `rxyz`, so "rotate 
                # around the x-axis first, then on the new y-axis, and finally 
                # on the new new z-axis", and the argument names are `ai`, `aj`, 
                # `ak` (i,j,k are generic placeholder names for the chosen axes), 
                # so the setting below performs a 45 degress rotation around 
                # (unmodified, since `ai=0` by default) y-axis.
                primary=(H.cm(z=-10), H.e2q(aj=45)),
                incremental=H.Incr.GLOBAL),
            H.posture( 5.0,
                primary=(H.cm(x=+10), H.e2q(ai=45)),
                incremental=H.Incr.GLOBAL),
            # Handy method for "keep previous posture for X seconds" message
            H.pause(2.0),
            # These three points are expressed as an increment to their previous
            # point, w.r.t. the (local) target frame, i.e. Yumi's left tooltip.
            # The local increments are internally converted to global and then 
            # applied to the previous point's global coordinates. 
            H.posture( 5.0,
                # [x y z]_WORLD + [0, +0.10, 0]_LOCAL
                secondary=(H.cm(y=+10), H.e2q(ak=-45)),
                incremental=H.Incr.LOCAL),
            H.posture( 5.0,
                # The rotation is computed along the gripper's y-axis
                secondary=(H.cm(z=-10), H.e2q(aj=45)),
                incremental=H.Incr.LOCAL),
            H.posture( 5.0,
                # A "screw-like" motion (a true screw motion cannot be achieved
                # using this helper class, as here we are just defining orientations, 
                # not trajectories; the "screw-like" effect is thus simply a cool 
                # byproduct)
                secondary=(H.cm(z=+20), H.e2q(ak=-180)),
                incremental=H.Incr.LOCAL),
    ])
