#!/usr/bin/env python3
import rospy

from yumift_msgs.helper import Helper as H


# In this example we illustrate further ways you can execute trajectories 
# described in custom frames other than `BASE`, `TOOL`, and `GLOBAL`.
# 
# Describing trajectories in custom frames can be extremely useful when motions
# need to be perfomed relative to a known object in the scene, or if the scene 
# itself changes.

if __name__ == "__main__":
    rospy.init_node("example_3_reference_frames_advanced", anonymous=True)
    
    # Add a custom frame relative to another frame. 
    # The parent frame can be any existing frame, even a custom one. 
    # The new frame will appear in the visualizer, and is essentially a 
    # ROS `tf`` transform.
    H.add_frame("table", H.cm(30, 0, 10), H.e2q(a1=180), H.FRAME_BASE)
    
    H.quick_send(
        topic="/trajectory",
        print_message="Reference frames and local coordinates",
        motion_default=H.Motion.INDIVIDUAL,
        postures=[
            # As usual, we use the `reference` argument to specify the how the 
            # trajectory controller should interpret the sent poses.
            # By using `LOCAL` and proving a frame name in `frame`, the given 
            # poses will be interpreted in the given frame.
            H.posture( 5.0,
                (H.cm(y=+15, z=10), H.e2q(), H.GRIP_OPENED),
                (H.cm(y=-25, z=5), H.e2q(), H.GRIP_OPENED),
                reference=H.Reference.LOCAL,
                frame="table"),
            H.pause(1.0),
            # The `TOOL` reference is actually just `LOCAL` with `frame=<gripper_fingers>`
            # (have a look at the visualizer to know the real name of the <...> placeholder).
            H.posture( 5.0,
                (H.e2q(a1=+30), H.GRIP_CLOSED),
                (H.e2q(a1=-30), H.GRIP_CLOSED),
                reference=H.Reference.TOOL),
            H.pause(1.0),
            # Instead, the `FRAME` reference interprets the poses as new definitions for 
            # the given `frame`, keeping the end-effectors in the same pose of in the 
            # new frame definition.
            H.posture( 5.0,
                (H.cm(y=+10, z=-10), H.e2q(a2=-15)),
                (H.cm(y=+10, z=-10), H.e2q(a2=-15)),  # TODO secondary is not used by controller
                reference=H.Reference.FRAME,
                frame="table"),
            H.pause(1.0),
            # The `GLOBAL` reference is actually just `FRAME` with `frame=<base>`
            # (have a look at the visualizer to know the real name of the <...> placeholder).
            H.posture( 5.0,
                (H.cm(x=+10, z=-20), H.e2q(a2=-30)),
                (H.cm(x=+10, z=-20), H.e2q(a2=-30)),
                reference=H.Reference.GLOBAL),
        ])
    