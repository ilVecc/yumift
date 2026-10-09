#!/usr/bin/env python3
import rospy

from yumift_msgs.helper import Helper as H


if __name__ == "__main__":
    rospy.init_node("example_6_robot_introspection", anonymous=True)
    
    posture = H.current_posture()
    print("Current right gripper frame (position, quat_wxyz)")
    print(posture.pose_gripper_r)
    
    H.quick_send(
        topic="/trajectory",
        print_message="Moving up to change position",
        motion_default=H.Motion.COORDINATED,
        postures=[
            H.posture( 5.0,
                (H.cm(x=45, z=20), H.e2q(a1=180)),
                None,
                H.Motion.COORDINATED),
        ])
    
    posture = H.current_posture()
    print("Right gripper frame after motion (position, quat_wxyz)")
    print(posture.pose_gripper_r)
        