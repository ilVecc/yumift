#!/usr/bin/env python3
import rospy

# This first example demostrates how to send trajectories to a Yumi trajectory
# controller accepting `yumift_msgs/YumiTrajectory` messages. 
# 
# To send a trajectory, you need to manually create a YumiTrajectory message,
# which in turn requires multiple YumiPosture messages. This is cumbersome and 
# error-prone. The Helper class provides useful methods that simplify the 
# creation of trajectories and postures.

from yumift_msgs.helper import Helper as H

def main():
    # First, we need to add ROS node capabilities to this Python script.
    # This essentially connects the script to the "ROS network".
    rospy.init_node("example_1_simple_trajectory", anonymous=True)

    # Then, simply define the pose of each arm as a tuple.
    # The first term is the position, the second the orientation, and the third
    # is the (optional) gripper width. Everything is expesssed with respect to 
    # the `yumi_base_link` reference frame.
    right_arm = (
        # End-effector position, as vector [x, y, z] in meters
        [0.35, -0.2, 0.2],
        # End-effector orientation, as quaterniorn [w, x, y, z]
        # Instead of a quaternion, you can use this handy function, which converts
        # Euler angles (with relative X-Y-Z convention by default) to quaternions.
        H.e2q(-90, 0, 0),
        # Gripper fingers width, in [mm]
        0.0
    )
    
    # Now, use the pose just created to create a posture message using the following
    # helper function. A posture is ideally a tuple composed of a right arm pose, a
    # left arm pose, and the time required to achieve them. 
    first_posture = H.posture(
        # time to get to this posture, in [s]
        5.0,
        # The pose defined above
        right_arm,
        # The compact version of the definition above, for the left arm this time
        # Notice the use of the `H.cm` function, which uses centimeters instead of
        # meters; alternatively, you can use `H.mm` for millimeters. 
        (H.cm(35, 20, 20), H.e2q(+90, 0, 0), 0.0)
    )

    # Finally, use this function to immediately create and send a list of postures
    # to a controller accepting YumiTrajectory messages.
    H.quick_send(
        topic="/trajectory",
        print_message="Example trajectory sent",
        wait_completion=True,   # sleep for the trajectory total time
        postures=[
            first_posture,
            H.posture( 5.0,
                (H.cm(35, -10, 4), H.e2q(0, 45, -135, "rzyx"), 20.0),
                (H.cm(35, +10, 4), H.e2q(0, 45, +135, "rzyx"), 20.0)),
            H.posture( 5.0,
                (H.cm(45, -15, 15), H.e2q(0, 180, 0)),  # no gripper value needed
                (H.cm(45, +15, 15), H.e2q(0, 180, 0))),
    ])


if __name__ == "__main__":
    main()
