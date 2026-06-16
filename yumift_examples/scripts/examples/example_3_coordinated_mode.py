#!/usr/bin/env python3
import rospy

from yumift_msgs.helper import Helper as H

# In this example we introduce a new way of instructing the robot to perform 
# two-handed tasks: COORDINATED mode.
# The fundamental difference from INDIVIDUAL mode is that the task is now defined
# with ABSOLUTE and RELATIVE positioning: the absolute frame is the "average pose" 
# of the right and left poses w.r.t. the base frame, while the relative frame is 
# the "difference pose" between the two w.r.t. the absolute frame.
# This allows to define two-hands coordination tasks (e.g. holding a tray, 
# twisting a bottle cap, folding a newspaper) with much more ease.

if __name__ == "__main__":
    rospy.init_node("example_3_coordinated_mode", anonymous=True)
    
    # `H.quick_send()` will create a ROS Publisher (always the same one, 
    # actually) multiple times, one for each trajectory you send.
    # Although this is essentially harmless, we can avoid these unnecessary 
    # initializations by first creating the publisher with `H.publisher()` 
    # and then using it as the `topic` parameter for `H.quick_send()`.
    pub = H.publisher("/trajectory")

    H.quick_send(
        topic=pub,
        print_message="Sent \"stretching\" trajectory in INDIVIDUAL mode",
        postures=[
            H.posture( 5.0, 
                # Already, we can see that individual mode is inefficient and 
                # cumbersome for highly symmetric tasks
                (H.cm(35, -20, 20), H.e2q(ai=180), 20.0),
                (H.cm(35, +20, 20), H.e2q(ai=180), 20.0)),
            H.posture( 2.0,
                # Now we want to close in the grippers, so we increase and decrease
                # right and left accordingly... and verbosily...
                # (extra: here we only want to update the positions, so we can skip 
                # the creation of a tuple to simplify the syntax a little)
                H.cm(y=+10), 
                H.cm(y=-10),
                incremental=H.Incr.GLOBAL),
            H.posture( 2.0,
                # Again, it is clear that we could exploit the symmetry of this
                # "stretching out" motion... 
                # Also, here we showcase incremental gripper closing, which adds 
                # the amount to the previous (20.0) gripper position
                (H.cm(y=-15, z=-16), -20.0),
                (H.cm(y=+15, z=-16), -20.0),
                incremental=H.Incr.GLOBAL),
            H.posture( 2.0,
                # Finally, incremental gripper opening (omitting tuple creation)
                # The amount is always added, so the + sign is just a reminder 
                # of what will happen: a positive increment
                +20.0,
                +20.0,
                incremental=H.Incr.GLOBAL),
    ])
    
    H.quick_send(
        topic=pub,
        print_message="Sent \"stretching\" trajectory in COORDINATED mode",
        # Now, we control with the "coordinated motion" mode
        mode=H.Mode.COORDINATED,
        postures=[
            H.posture( 5.0, 
                # Primary is now the ABSOLUTE frame, so the "average pose" of 
                # the two grippers w.r.t. the base frame.
                # Simply, we want the grippers at this average location, and both
                # the fingers to be opened at 20mm.
                primary=(H.cm(x=35, z=20), H.e2q(ai=180), 20),
                # Secondary is now the RELATIVE frame, so the "difference pose" 
                # of the two grippers w.r.t. the absolute frame.
                # Again, simply, we want the grippers to be this far from each other.
                secondary=(H.cm(y=40), H.e2q()),
                # Use the COORDINATED mode per-posture as well
                mode=H.Mode.COORDINATED),
            H.posture( 2.0,
                # No absolute request...
                None,  # or tuple() can be used to avoid keyword assignment
                # ... only a relative inwards displacement of the grippers
                H.cm(y=-20),
                H.Mode.COORDINATED, 
                H.Incr.GLOBAL),
            H.posture( 2.0, 
                # Now, we want both grippers to move down and close their fingers...
                (H.cm(z=-16), -20),
                # ... while moving outwards.
                H.cm(y=+30), 
                H.Mode.COORDINATED, 
                H.Incr.GLOBAL),
            H.posture( 2.0,
                # And finally to open both grippers' fingers again.
                +20.0,
                None,
                H.Mode.COORDINATED, 
                H.Incr.GLOBAL),
    ])
    
    #
    # Here are a couple more coordinated motions to exemplify the APIs.
    #
    
    H.quick_send(
        topic=pub,
        print_message="Sent more coordinated motions",
        mode=H.Mode.COORDINATED,
        postures=[
            H.posture(5.0,
                (H.cm(40, 0, 20), H.e2q(ai=180)),
                (H.cm(y=10), H.e2q()),
                H.Mode.COORDINATED),
            H.posture(5.0,
                # notice the use of `ai=180` since we want to keep the absolute 
                # frame facing down, and on top of that add a 45deg rotation on 
                # the new y-axis (this of course can be simplified using the 
                # handy incremental keyword)
                (H.cm(40, 0, 20), H.e2q(ai=180, aj=45)),
                (H.cm(y=10), H.e2q()),
                H.Mode.COORDINATED),
            H.posture( 5.0,
                H.e2q(aj=-45),
                H.cm(y=+15),
                H.Mode.COORDINATED,
                H.Incr.LOCAL),
            H.posture( 5.0,
                H.cm(y=+10, z=+20),
                None,
                H.Mode.COORDINATED,
                H.Incr.GLOBAL),
            H.posture( 5.0,
                H.cm(y=-20, z=-40),
                H.cm(y=-15),
                H.Mode.COORDINATED,
                H.Incr.GLOBAL)
    ])

    H.quick_send(
        topic=pub,
        print_message="Sent a final coordinated motion parade!",
        mode=H.Mode.COORDINATED,
        postures = [
            H.posture( 8.0,
                (H.cm(40, 0, 20), H.e2q(ai=180)),
                (H.cm(y=25), H.e2q()),
                H.Mode.COORDINATED),
            H.posture( 8.0,
                H.e2q(ai=+15),
                H.e2q(aj=+45),
                H.Mode.COORDINATED,
                H.Incr.LOCAL),
            H.posture( 8.0,
                H.e2q(ai=-30),
                H.e2q(aj=-45),
                H.Mode.COORDINATED,
                H.Incr.LOCAL),
            H.posture( 8.0,
                H.e2q(ai=+15),
                H.e2q(ai=+30),
                H.Mode.COORDINATED,
                H.Incr.LOCAL),
            H.posture( 8.0,
                H.e2q(ak=40),
                H.e2q(ai=-30),
                H.Mode.COORDINATED,
                H.Incr.LOCAL),
            H.posture( 8.0,
                H.e2q(ak=-80),
                H.e2q(ak=+45),
                H.Mode.COORDINATED,
                H.Incr.LOCAL),
            H.posture( 8.0,
                H.e2q(ak=40),
                H.e2q(ak=-45),
                H.Mode.COORDINATED,
                H.Incr.LOCAL),
    ])
