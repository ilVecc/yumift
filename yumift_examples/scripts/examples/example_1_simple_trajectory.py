#!/usr/bin/env python3
import rospy

from yumift_msgs.msg import YumiTrajectory
from yumift_msgs.yumi_posture_helper import Helper


def main():
    # starting ROS node and subscribers
    rospy.init_node("example_1_simple_trajectory", anonymous=True)
    pub = rospy.Publisher("/trajectory", YumiTrajectory, queue_size=1, latch=True)
    rospy.sleep(0.1)

    msg = Helper.trajectory(
        YumiTrajectory.INDIVIDUAL,              # trajectory control mode (more on this in following examples)
        [
        Helper.posture(
            5.0,                                # time to get to this point [s]
            (                                   # `yumi_base_link` is the reference frame
                [0.35, -0.2, 0.2],              # right arm position, as vector [x, y, z]
                Helper.eul2quat(-90,0,0),       # right arm orientation, as quaterniorn [w, x, y, z]
                0.0                             # gripper width for the right fingers [mm]
            ),
            ([0.35, +0.2, 0.2], Helper.eul2quat(+90,0,0), 0.0)), # compact definition for left arm
        Helper.posture(
            5.0,
            ([0.35, -0.1, 0.04], Helper.eul2quat(0, 45, -135, "rzyx"), 20.0),
            ([0.35, +0.1, 0.04], Helper.eul2quat(0, 45, +135, "rzyx"), 20.0)),
        Helper.posture(
            5.0,
            ([0.45, -0.15, 0.15], Helper.eul2quat(0,180,0)),
            ([0.45, +0.15, 0.15], Helper.eul2quat(0,180,0))),
        ])
    pub.publish(msg)
    print("Trajectory message sent")
    rospy.sleep(15+1)                           # sleep for at least the total time


if __name__ == "__main__":
    main()
