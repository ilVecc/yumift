#!/usr/bin/env python3
import rospy

from yumift_msgs.helper import Helper as H

# This second example is a follow-up to the first one and demostrates how to 
# send trajectories in a different reference frame.
# 
# As seen previously, a trajectory is a list of global postures with a duration. 
# Sometime, it's more useful to express the next point along the trajectory as a 
# movement from the previous one, or with respet to a specific reference frame, 
# so to avoid manually computing each posture.
# This can be conveniently achieved with the `reference` keyword.

if __name__ == "__main__":
    # First, initialize the ROS node
    rospy.init_node("example_2_reference_frames_basic", anonymous=True)
    
    # Then, as seen in the previous example, quickly contruct and send a trajectory
    H.quick_send(
        topic="/trajectory",
        print_message="Reference-specific trajectory sent",
        wait_completion=True,
        postures=[
            # Let's start in a comfortable position for the demo.
            H.posture( 8.0,
                primary=(H.cm(35, -20, 20), H.e2q(-90, 0, 0)),
                secondary=(H.cm(35, +20, 20), H.e2q(+90, 0, 0)),
                reference=H.Reference.BASE),
            # This handy method creates a "keep previous posture for X seconds" 
            # message and, for now, it uses magic to do that. We will use it to 
            # pause and understand the various motions from now on.
            H.pause(2.0),
            ###################################################################
            # The following three postures are expressed as an "increment" from 
            # the previous one w.r.t. the (global) world frame, i.e. Yumi's base.
            # Essentially, we are now passing a transformation (translation and 
            # rotation) instead of a pose (position and orientation), and this 
            # transformation acts on the world frame. Effectively, this performs
            # a motion as if we were modifying the world frame.
            H.posture( 5.0,
                # Position is a vector to be added to the initial position as
                #   [0.35, -0.2, 0.2]_WORLD + [-0.15, 0, 0]_WORLD
                # For simplicity, no explicit rotation is defined now, meaning no 
                # rotation will be applied on the current orientation.
                primary=(H.cm(-15, 0, 0),),
                # No explicit secondary pose provided, defaulting to "do nothing".
                # This argument can also be omitted entirely.
                secondary=tuple(),
                # Finally, explicitely set the reference to be the world frame.
                # If you forget to do this, the arguments above are interpreted 
                # as a desired pose instead of a desired transformation.
                reference=H.Reference.GLOBAL),
            H.pause(2.0),
            # To simplify the commands, you can avoid writing the coordinates 
            # you do not care about; this defaults them to 0.
            # For orientations, the default behaviour is `rxyz`, so "rotate 
            # around the x-axis first, then on the new y-axis, and finally 
            # on the new new z-axis", and the argument names are `a1`, `a2`, 
            # `a3` (generic placeholder names for the chosen order).
            # The setting below thus performs a 45 degress rotation around the 
            # unrotated z-axis (since `a1=0` and `a2=0` by default).
            # It is important to remember that GLOBAL increments are expressed 
            # in the world frame, meaning that an z-axis rotation pivots the 
            # gripper around the world z-axis. Thus, z-axis translations do 
            # not interfere with z-axis rotations (and any other same-axis 
            # pair) ...
            H.posture( 5.0,
                primary=(H.cm(z=-10), H.e2q(a3=45)),
                reference=H.Reference.GLOBAL),
            H.pause(2.0),
            # ... but different-axis pairs (or any multi-axis transformation, so 
            # in general any other transformation) can result in apparently strange 
            # behaviour. The example below moves the y-axis and rotates around the
            # x-axis (we pedagogically chose to change the axes order to "relative,
            # first y, then new z, then new new x" so `a3` in this case defines the 
            # angle around the "new new x axis", which is simply the original x-axis).
            # This achieves a seemingly unnatural motion, easily explanied by 
            # remembering that the global transform essentially locks the gripper's 
            # pose to the world frame, and then changes the world frame as desired.
            H.posture( 5.0,
                primary=(H.cm(y=-20), H.e2q(a3=-90, axes="ryzx")),
                reference=H.Reference.GLOBAL),
            # Again, wait a little before the next motion.
            # Now we can explain what a "pause" command actually is: a simple global
            # increment by a zero translation and an identity rotation!
            H.pause(2.0),
            ###################################################################
            # The following three postures are expressed as an "increment" to the 
            # previous w.r.t. the (tool) target frame, i.e. Yumi's tooltips.
            # These poses are expressed using the previous tool pose.
            H.posture( 5.0,
                # [x y z]_TOOL + [0, +0.10, 0]_TOOL
                secondary=(H.cm(y=+10), H.e2q(a3=-45)),
                reference=H.Reference.TOOL),
            H.posture( 5.0,
                # The rotation is computed along the gripper's y-axis
                secondary=(H.cm(z=-10), H.e2q(a2=45)),
                reference=H.Reference.TOOL),
            H.posture( 5.0,
                # A "screw-like" motion 
                # (A true screw motion cannot be achieved using this helper class, 
                # as here we are just defining orientations, not trajectories; the 
                # "screw-like" effect is thus simply a cool byproduct)
                secondary=(H.cm(z=+20), H.e2q(a3=-180)),
                reference=H.Reference.TOOL),
    ])
