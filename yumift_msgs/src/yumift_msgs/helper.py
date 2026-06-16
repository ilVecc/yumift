from typing import List, Union, Optional, Tuple
from enum import Enum

import rospy
import tf.transformations as trans

import numpy as np
import quaternion as quat
from numpy.typing import ArrayLike

from dynamicals.utils import Frame

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
    def e2q(ai : float = 0, aj : float = 0, ak : float = 0, axes : str = "rxyz") -> np.ndarray:
        """ Create quaternions from Euler angles, in either relative or static base. """
        return np.roll(trans.quaternion_from_euler(np.radians(ai), np.radians(aj), np.radians(ak), axes), 1)
    
    @staticmethod
    def mm(x: float = 0, y: float = 0, z: float = 0) -> np.ndarray:
        """ Create positions from millimeters. """
        return np.array([x,y,z]) * 0.001
    
    @staticmethod
    def cm(x: float = 0, y: float = 0, z: float = 0) -> np.ndarray:
        """ Create positions from centimeters. """
        return np.array([x,y,z]) * 0.01
    
    @staticmethod
    def _parse_input(point : Optional[Union[tuple, ArrayLike]]):
        # force tuple to keep the parsing clean
        if point is None:
            point = tuple()
        if not isinstance(point, tuple):
            point = (point,)
        
        # check elements in the tuple, and prepare their sizes
        sizes = np.array([np.size(o) for o in point])
        if len(sizes) > 4:
            raise Exception("Too many parameters")
        
        # filter component (grip, pos, rot, vel) by vector size
        idx = np.where(sizes == 1)[0]
        grip = np.array(point[idx[0]]) if len(idx) == 1 else np.nan

        idx = np.where(sizes == 3)[0]
        pos = np.array(point[idx[0]]) if len(idx) == 1 else np.nan*np.ones(3)

        idx = np.where(sizes == 4)[0]
        rot = np.array(point[idx[0]]) if len(idx) == 1 else np.nan*np.ones(4)

        idx = np.where(sizes == 6)[0]
        vel = np.array(point[idx[0]]) if len(idx) == 1 else np.nan*np.ones(6)

        return grip, pos, rot, vel

    @staticmethod
    def posture( 
        duration : float = 0.0, 
        primary : Optional[Union[tuple, ArrayLike]] = tuple(), 
        secondary : Optional[Union[tuple, ArrayLike]] = tuple(),
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
        
        # if wait_time == False, then T = 0  (no sleep)
        if not (isinstance(wait_completion, bool) and not wait_completion):
            if postures is None:
                # TODO assuming all routines take 5 seconds is bad
                time = 5.0
            else:
                # if wait_time <  0.0,   then T = |wait_time|
                if isinstance(wait_completion, float) and wait_completion < 0:
                    time = abs(wait_completion)
                else:
                    # if wait_time == True,  then T = sum(durations) 
                    time = sum([p.time_to_execute.to_sec() for p in postures])
                    # if wait_time >= 0.0,   then T = sum(durations) + wait_time
                    if isinstance(wait_completion, float) and wait_completion > 0:
                        time += abs(wait_completion)
            wait_completion = True  # ensure bool type
        
        # prepare the message
        msg = YumiTrajectory()
        if postures is None:
            msg.mode = YumiTrajectory.ROUTINE
            msg.routine_name = routine_name
        else:
            msg.mode = mode.value[0]
            msg.trajectory = postures
        
        # send the message
        msg.header.stamp = rospy.Time.now()
        pub.publish(msg)
        
        # print if required
        if print_message is not None:
            rospy.loginfo(print_message)
        
        # wait if required
        if wait_completion:    
            rospy.sleep(time)

    @staticmethod
    def decode_trajectory(init_posture : Tuple[Frame, Frame, float, float], traj_msg : YumiTrajectory) -> List[Tuple[Tuple[Frame, Frame, float, float], float]]:
        trajectory = [(init_posture, 0)]
        # append trajectory points from msg
        for posture in traj_msg.trajectory:
            posture: YumiPosture
            pos_1 = Helper.sanitize_pos(posture.pose_primary.position, default_none=False)
            rot_1 = Helper.sanitize_rot(posture.pose_primary.orientation, default_none=False)
            vel_1 = Helper.sanitize_vel(posture.twist_primary)
            frame_1 = Frame(pos_1, rot_1) #, vel_1)
            pos_2 = Helper.sanitize_pos(posture.pose_secondary.position, default_none=False)
            rot_2 = Helper.sanitize_rot(posture.pose_secondary.orientation, default_none=False)
            vel_2 = Helper.sanitize_vel(posture.twist_secondary)
            frame_2 = Frame(pos_2, rot_2) #, vel_2)
            grip_r = Helper.sanitize_grip(posture.gripper_right, default_none=False)
            # TODO use me more for other things as well
            # set the same grip width if in coordinated mode
            if posture.mode != YumiPosture.COORDINATED:
                grip_l = Helper.sanitize_grip(posture.gripper_left, default_none=False)
            else:
                grip_l = grip_r
            duration = posture.time_to_execute.to_sec()
            
            # convert everything to GLOBAL COORDINATES
            if posture.incremental != YumiPosture.OFF:
                prev_1, prev_2, prev_grip_r, prev_grip_l = trajectory[-1][0]
                # TODO remove .motionless() and handle twist (can be None) in incremental mode
                prev_1 = prev_1.motionless()
                prev_2 = prev_2.motionless()
                grip_r = grip_r + prev_grip_r
                grip_l = grip_l + prev_grip_l
                
                # handle incremental postures
                if posture.incremental == YumiPosture.LOCAL:
                    # next_posture = prev_posture @ local_transformation
                    frame_1 = prev_1 @ frame_1
                    frame_2 = prev_2 @ frame_2
                elif posture.incremental == YumiPosture.GLOBAL:
                    # next_posture = global_transformation @ prev_posture
                    frame_1 = frame_1 @ prev_1
                    frame_2 = frame_2 @ prev_2
                else:
                    rospy.logerr(f"Unknown incremental mode {posture.incremental}")
            
            frame_1.vel = vel_1
            frame_2.vel = vel_2
            trajectory.append(((frame_1, frame_2, grip_r, grip_l), duration))
        return trajectory

    @staticmethod
    def sanitize_grip(grip: float, default_none: bool = True) -> Optional[float]:
        if np.isnan(grip):
            return None if default_none else 0.
        return grip

    @staticmethod
    def sanitize_pos(pos: Point, default_none: bool = True) -> Optional[np.ndarray]:
        pos = np.array([pos.x, pos.y, pos.z])
        if np.any(np.isnan(pos)):
            return None if default_none else np.zeros(3)
        return pos

    @staticmethod
    def sanitize_rot(ori: Quaternion, default_none: bool = True) -> Optional[np.quaternion]:
        ori = quat.quaternion(ori.w, ori.x, ori.y, ori.z)
        if ori == quat.zero or ori.isnan():
            return None if default_none else quat.one
        return ori

    @staticmethod
    def sanitize_vel(vel: Twist, default_none: bool = True) -> Optional[np.ndarray]:
        vel = np.array([vel.linear.x, vel.linear.y, vel.linear.z,
                        vel.angular.x, vel.angular.y, vel.angular.z])
        if np.any(np.isnan(vel)):
            return None if default_none else np.zeros(6)
        return vel
