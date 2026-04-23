from typing import List, Union, Optional
from enum import Enum

import rospy
import tf.transformations as trans

import numpy as np

from geometry_msgs.msg import Point, Quaternion, Pose, Twist
from yumift_msgs.msg import YumiPosture, YumiTrajectory



class Helper():
    
    class Mode(Enum):
        INDIVIDUAL  = YumiTrajectory.INDIVIDUAL,   YumiPosture.INDIVIDUAL
        RIGHT       = YumiTrajectory.RIGHT,        YumiPosture.RIGHT
        LEFT        = YumiTrajectory.LEFT,         YumiPosture.LEFT
        COORDINATED = YumiTrajectory.COORDINATED,  YumiPosture.COORDINATED
        ABSOLUTE    = YumiTrajectory.ABSOLUTE,     YumiPosture.ABSOLUTE
        RELATIVE    = YumiTrajectory.RELATIVE,     YumiPosture.RELATIVE

    class Incr(Enum):
        OFF     = YumiPosture.OFF
        LOCAL   = YumiPosture.LOCAL
        GLOBAL  = YumiPosture.GLOBAL

    @staticmethod
    def e2q(ai : float = 0, aj : float = 0, ak : float = 0, axes : str = "rxyz"):
        return np.roll(trans.quaternion_from_euler(np.radians(ai), np.radians(aj), np.radians(ak), axes), 1)
    
    @staticmethod
    def mm(x: float = 0, y: float = 0, z: float = 0):
        return np.array([x,y,z]) * 0.001
    
    @staticmethod
    def cm(x: float = 0, y: float = 0, z: float = 0):
        return np.array([x,y,z]) * 0.01
    
    @staticmethod
    def _parse_input(point : tuple):
        sizes = np.array([np.size(o) for o in point])
        if len(sizes) > 4:
            raise Exception("Too many parameters")

        idx = np.where(sizes == 1)[0]
        grip = np.array(point[idx[0]]) if len(idx) == 1 else 0.0

        idx = np.where(sizes == 3)[0]
        pos = np.array(point[idx[0]]) if len(idx) == 1 else np.zeros(3)

        idx = np.where(sizes == 4)[0]
        rot = np.array(point[idx[0]]) if len(idx) == 1 else np.array([1,0,0,0])

        idx = np.where(sizes == 6)[0]
        vel = np.array(point[idx[0]]) if len(idx) == 1 else np.zeros(6)

        return grip, pos, rot, vel

    @staticmethod
    def posture( 
        duration : float = 0.0, 
        primary : tuple = tuple(), 
        secondary : tuple = tuple(),
        mode : Mode = Mode.INDIVIDUAL, 
        incremental : Incr = Incr.OFF
    ):
        grip_r, pos_1, rot_1, vel_1 = Helper._parse_input(primary)
        grip_l, pos_2, rot_2, vel_2 = Helper._parse_input(secondary)
        
        posture = YumiPosture(
            pose_primary = Pose(
                position = Point(x=pos_1[0], y=pos_1[1], z=pos_1[2]),
                orientation = Quaternion(w=rot_1[0], x=rot_1[1], y=rot_1[2], z=rot_1[3])),
            pose_secondary = Pose(
                position = Point(x=pos_2[0], y=pos_2[1], z=pos_2[2]),
                orientation = Quaternion(w=rot_2[0], x=rot_2[1], y=rot_2[2], z=rot_2[3])),
            twist_primary = Twist(
                linear=Point(x=vel_1[0], y=vel_1[1], z=vel_1[2]),
                angular=Point(x=vel_1[3], y=vel_1[4], z=vel_1[5])),
            twist_secondary = Twist(
                linear=Point(x=vel_2[0], y=vel_2[1], z=vel_2[2]),
                angular=Point(x=vel_2[3], y=vel_2[4], z=vel_2[5])),
            gripper_right = grip_r,
            gripper_left  = grip_l,
            time_to_execute = rospy.Duration(secs=duration),
            mode = mode.value[1],
            incremental = incremental.value)
        
        return posture

    @staticmethod
    def pause(time : float):
        posture = YumiPosture(
            mode=YumiPosture.INDIVIDUAL,
            incremental=YumiPosture.GLOBAL, 
            time_to_execute=rospy.Duration.from_sec(time))
        posture.pose_primary.orientation.w = 1
        posture.pose_secondary.orientation.w = 1
        return posture
    
    @staticmethod
    def trajectory(mode : Mode = Mode.INDIVIDUAL, postures : List[YumiPosture] = []):
        msg = YumiTrajectory()
        msg.mode = mode.value[0]
        msg.trajectory = postures
        msg.header.stamp = rospy.Time.now()
        return msg
    
    @staticmethod
    def publisher(topic : str = "/trajectory"):
        # `latch=True` ensures that early messages get sent to the subscribers: 
        # subscribers need some time to be notified of a new publisher on their 
        # same topic, so messages sent immediately after the initialization of 
        # the new publisher are not always received. Latching caches the last 
        # sent message (which is also the only one in our case), so that when 
        # subscribers get notified they also get the message.   
        return rospy.Publisher(topic, YumiTrajectory, queue_size=1, latch=True)
    
    @staticmethod
    def quick_send(
        topic : Union[str, rospy.Publisher] = "/trajectory",
        mode : Mode = Mode.INDIVIDUAL,
        postures : Optional[List[YumiPosture]] = None,
        routine_name : Optional[str] = None,
        print_message : Optional[str] = None,
        wait_completion : Union[bool, float] = True,
    ):
        """ Quickly send a trajectory or a routine.
        """
        
        # if string, create a publisher for that topic, 
        # otherwise use the provided publisher
        pub = Helper.publisher(topic) if isinstance(topic, str) else topic
        
        if postures is None and routine_name is None:
            raise Exception("No trajectory/routine specified")

        msg = YumiTrajectory()
        
        if postures is None:
            msg.mode = YumiTrajectory.ROUTINE
            msg.routine_name = routine_name
        else:
            msg.mode = mode.value[0]
            msg.trajectory = postures
        
        msg.header.stamp = rospy.Time.now()
        pub.publish(msg)
        
        if print_message is not None:
            rospy.loginfo(print_message)
        
        if isinstance(wait_completion, float):
            extra_time = abs(wait_completion)
            wait_completion = True
        else:
            extra_time = 0
        
        if wait_completion is True:
            if postures is None:
                rospy.sleep(5.0)  # default sleep time for a routine
            else:
                rospy.sleep(sum([p.time_to_execute.to_sec() for p in postures]) + extra_time)
