import rospy
import tf.transformations as trans

from geometry_msgs.msg import Point, Quaternion, Twist
from yumift_msgs.msg import YumiPosture, YumiTrajectory

import numpy as np
from typing import List, Union


class Helper():

    @staticmethod
    def eul2quat(ai : float, aj : float, ak : float, axes : str = "rxyz"):
        return np.roll(trans.quaternion_from_euler(np.radians(ai), np.radians(aj), np.radians(ak), axes), 1)

    @staticmethod
    def _parse_input(point : tuple):
        sizes = np.array([np.size(o) for o in point])
        if len(sizes) > 4:
            raise Exception("Too many parameters")

        idx = np.where(sizes == 1)[0]
        grip = np.array(point[idx[0]]) if len(idx) == 1 else None

        idx = np.where(sizes == 3)[0]
        pos = np.array(point[idx[0]]) if len(idx) == 1 else None

        idx = np.where(sizes == 4)[0]
        rot = np.array(point[idx[0]]) if len(idx) == 1 else None

        idx = np.where(sizes == 6)[0]
        vel = np.array(point[idx[0]]) if len(idx) == 1 else None

        return grip, pos, rot, vel

    @staticmethod
    def posture( 
        duration : float = 0.0, 
        primary : tuple = tuple(), 
        secondary : tuple = tuple(),
        mode : int = YumiPosture.INDIVIDUAL, 
        incremental : int = YumiPosture.OFF
    ):
        grip_r, pos_1, rot_1, vel_1 = Helper._parse_input(primary)
        grip_l, pos_2, rot_2, vel_2 = Helper._parse_input(secondary)
        
        posture = YumiPosture(
            gripper_right = grip_r,
            gripper_left  = grip_l,
            time_to_execute = rospy.Duration(secs=duration),
            mode = mode,
            incremental = incremental)
        
        if pos_1 is not None:
            posture.pose_primary.position = Point(x=pos_1[0], y=pos_1[1], z=pos_1[2])
        if rot_1 is not None:
            posture.pose_primary.orientation = Quaternion(w=rot_1[0], x=rot_1[1], y=rot_1[2], z=rot_1[3])
        if vel_1 is not None:
            posture.twist_primary = Twist(
                linear=Point(x=vel_1[0], y=vel_1[1], z=vel_1[2]),
                angular=Point(x=vel_1[3], y=vel_1[4], z=vel_1[5]))
        if pos_1 is not None:
            posture.pose_secondary.position = Point(x=pos_2[0], y=pos_2[1], z=pos_2[2])
        if rot_1 is not None:
            posture.pose_secondary.orientation = Quaternion(w=rot_2[0], x=rot_2[1], y=rot_2[2], z=rot_2[3])
        if vel_1 is not None:
            posture.twist_secondary = Twist(
                linear=Point(x=vel_2[0], y=vel_2[1], z=vel_2[2]),
                angular=Point(x=vel_2[3], y=vel_2[4], z=vel_2[5]))

        return posture
    
    @staticmethod
    def trajectory(mode : int = YumiTrajectory.INDIVIDUAL, postures : List[YumiPosture] = []):
        msg = YumiTrajectory()
        msg.mode = mode
        msg.trajectory = postures
        msg.header.stamp = rospy.Time.now()
        return msg
    
    def quick_send(
        topic : str = "/trajectory", 
        postures : Union[List[YumiPosture], None] = None,
        routine_name : Union[str, None] = None,
        print_message : Union[str, None] = None,
        wait_completion : bool = True,
    ):
        """ Quickly send an INDIVIDUAL mode trajectory or a routine.
        """
        pub = rospy.Publisher(topic, YumiTrajectory, queue_size=1, latch=True)
        
        if postures is None and routine_name is None:
            raise Exception("No trajectory/routine specified")

        msg = YumiTrajectory()
        
        if postures is None:
            msg.mode = YumiTrajectory.ROUTINE
            msg.routine_name = routine_name
        else:
            msg.mode = YumiTrajectory.INDIVIDUAL
            msg.trajectory = postures
        
        msg.header.stamp = rospy.Time.now()
        pub.publish(msg)
        
        if print_message is not None:
            rospy.loginfo(print_message)
        
        if wait_completion is True:
            if postures is None:
                rospy.sleep(5.0)  # default sleep time for a routine
            else:
                rospy.sleep(sum([p.time_to_execute for p in postures]))
    