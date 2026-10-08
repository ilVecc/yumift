from ctypes import ArgumentError
from typing import List, Union, Optional, Tuple
from enum import Enum

import rospy
import tf2_ros
import tf.transformations as trans

import numpy as np
import quaternion as quat
from numpy.typing import ArrayLike

from dynamicals.utils import Frame

from geometry_msgs.msg import Point, Quaternion, Pose, Twist, TransformStamped
from yumift_msgs.msg import YumiPosture, YumiTrajectory


buffer = tf2_ros.Buffer()
static_broadcaster = tf2_ros.StaticTransformBroadcaster()

class Helper():
    
    class Motion(Enum):
        INDIVIDUAL  = YumiTrajectory.INDIVIDUAL,   YumiPosture.MOTION_INDIVIDUAL
        RIGHT       = YumiTrajectory.RIGHT,        YumiPosture.MOTION_RIGHT
        LEFT        = YumiTrajectory.LEFT,         YumiPosture.MOTION_LEFT
        COORDINATED = YumiTrajectory.COORDINATED,  YumiPosture.MOTION_COORDINATED
        ABSOLUTE    = YumiTrajectory.ABSOLUTE,     YumiPosture.MOTION_ABSOLUTE
        RELATIVE    = YumiTrajectory.RELATIVE,     YumiPosture.MOTION_RELATIVE

    class Reference(Enum):
        BASE   = YumiPosture.REFERENCE_BASE
        TOOL   = YumiPosture.REFERENCE_TOOL
        LOCAL  = YumiPosture.REFERENCE_LOCAL
        GLOBAL = YumiPosture.REFERENCE_GLOBAL
        FRAME  = YumiPosture.REFERENCE_FRAME
    
    FRAME_DEFAULT = "yumi_base_link"
    GRIP_MIN = 0
    GRIP_MAX = 25
    
    
    ###########################################################################
    #                          CLIENT-SIDE METHODS                            #
    ###########################################################################
    
    @staticmethod
    def _quat_xyzw_to_wxyz(quat_xyzw):
        return np.roll(quat_xyzw, 1)
    
    @staticmethod
    def _quat_wxyz_to_xyzw(quat_wxyz):
        return np.roll(quat_wxyz, -1)
    
    @staticmethod
    def e2q(a1 : float = 0, a2 : float = 0, a3 : float = 0, axes : str = "rxyz") -> np.ndarray:
        """ Create quaternions from Euler angles, in either relative or static base. 
            `a1`, `a2`, `a3` are the angles for the chosen order in `axes`.
            The output quaternion follows the `[w, x, y, z]` convention.
        """
        return Helper._quat_xyzw_to_wxyz(trans.quaternion_from_euler(np.radians(a1), np.radians(a2), np.radians(a3), axes))
    
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
        motion : Motion = Motion.INDIVIDUAL, 
        reference : Reference = Reference.BASE,
        frame : str = FRAME_DEFAULT
    ):
        """ Create a `YumiPosture` message.
            :param duration: time to execute this posture
            :param primary: position, rotation, velocity and grip for the RIGHT arm or the ABSOLUTE pose (grip is set to both gripper in this case)
            :param primary: position, rotation, velocity and grip for the LEFT arm or the RELATIVE pose (grip is ignored in this case)
            :param motion: how to interpret `primary` and `secondary`
            :param reference: where to interpret `primary` and `secondary`
            :param frame: reference frame of `primary` and `secondary` (RELATIVE pose is always expressed wrt the ABSOLUTE pose)
        """
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
            motion = motion.value[1],
            reference = reference.value,
            ref_frame = frame
        )
        
        return posture

    @staticmethod
    def pause(time : float):
        posture = YumiPosture(
            motion=YumiPosture.MOTION_INDIVIDUAL,
            reference=YumiPosture.REFERENCE_GLOBAL, 
            time_to_execute=rospy.Duration.from_sec(time))
        posture.pose_primary.orientation.w = 1
        posture.pose_secondary.orientation.w = 1
        return posture
    
    @staticmethod
    def grip(grip_r : float = GRIP_MIN, grip_l : float = GRIP_MIN):
        posture = YumiPosture(
            gripper_right=grip_r,
            gripper_left=grip_l,
            motion=YumiPosture.MOTION_INDIVIDUAL,
            reference=YumiPosture.REFERENCE_GLOBAL, 
            time_to_execute=rospy.Duration.from_sec(0.9))
        posture.pose_primary.orientation.w = 1
        posture.pose_secondary.orientation.w = 1
        return posture
    
    @staticmethod
    def release(grip_r : float = GRIP_MAX, grip_l : float = GRIP_MAX):
        posture = YumiPosture(
            gripper_right=grip_r,
            gripper_left=grip_l,
            motion=YumiPosture.MOTION_INDIVIDUAL,
            reference=YumiPosture.REFERENCE_GLOBAL, 
            time_to_execute=rospy.Duration.from_sec(0.9))
        posture.pose_primary.orientation.w = 1
        posture.pose_secondary.orientation.w = 1
        return posture
    
    @staticmethod
    def trajectory(motion_override : Motion = Motion.INDIVIDUAL, postures : List[YumiPosture] = []):
        msg = YumiTrajectory()
        msg.mode = motion_override.value[0]
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
        motion_override : Motion = Motion.INDIVIDUAL,
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
            msg.mode = motion_override.value[0]
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
    def add_frame(name : str, pos, rot, parent : str = FRAME_DEFAULT, replace = True):
        """ Set a static (unchanging) frame attached to another existing frame.
            :param pos: position wrt `parent` (in meters)
            :param rot: rotation wrt `parent` (wxyz quaternion)
            :param name: name of the new frame (must not exist in `tf`)
            :param parent: name of the parent frame (must exist in `tf`)
        """
        
        # if not buffer.can_transform(FRAME_DEFAULT, parent, rospy.Time(0)):
        #     rospy.logerr(f"Parent frame \"{parent}\" does not exist. Aborting.")
        #     return False
        # if buffer.can_transform(FRAME_DEFAULT, name, rospy.Time(0)):
        #     if not replace:
        #         rospy.logerr(f"Frame name \"{name}\" already in use. Aborting.")
        #         return False
        #     rospy.logwarn(f"Frame name \"{name}\" already in use. Replacing it.")
            
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = parent
        t.child_frame_id = name
        t.transform.translation.x = pos[0]
        t.transform.translation.y = pos[1]
        t.transform.translation.z = pos[2]
        t.transform.rotation.w = rot[0]
        t.transform.rotation.x = rot[1]
        t.transform.rotation.y = rot[2]
        t.transform.rotation.z = rot[3]
        static_broadcaster.sendTransform(t)
        
        return True


    ###########################################################################
    #                        CONTROLLER-SIDE METHODS                          #
    ###########################################################################


    @staticmethod
    def decode_trajectory(init_posture : Tuple[Frame, Frame, float, float], traj_msg : YumiTrajectory) -> List[Tuple[Tuple[Frame, Frame, float, float], float]]:
        """ Decode the trajectory into numpy objects, and trasform all poses into base frame, 
            while retaining info regarding the MOTION type.
        """
        
        # TODO move this out
        transform_listener = tf2_ros.TransformListener(buffer)
        rospy.sleep(0.5)
        
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
            if posture.motion != YumiPosture.MOTION_COORDINATED:
                grip_l = Helper.sanitize_grip(posture.gripper_left, default_none=False)
            else:
                grip_l = grip_r
            duration = posture.time_to_execute.to_sec()
            
            
            # convert everything to GLOBAL COORDINATES (base)
            if posture.reference != YumiPosture.REFERENCE_BASE:
                prev_1, prev_2, prev_grip_r, prev_grip_l = trajectory[-1][0]
                # TODO remove .motionless() and handle twist (can be None) in the right reference
                prev_1 = prev_1.motionless()
                prev_2 = prev_2.motionless()
                # TODO introduce this again?
                # grip_r = grip_r + prev_grip_r
                # grip_l = grip_l + prev_grip_l
                
                # resolve the unknown "ref_frame" frame if present
                if posture.reference in [YumiPosture.REFERENCE_LOCAL, YumiPosture.REFERENCE_FRAME]:
                    
                    if posture.ref_frame == "" or not buffer.can_transform(Helper.FRAME_DEFAULT, posture.ref_frame, rospy.Time(0)):
                        error_string = f"Reference frame \"{posture.ref_frame}\" does not exist."
                        rospy.logerr(error_string)
                        raise ArgumentError(error_string)
                    
                    # construct frame-to-base (bHf) or new_frame-to-frame (fHf') transformation
                    T = buffer.lookup_transform(Helper.FRAME_DEFAULT, posture.ref_frame, rospy.Time(0))
                    t, r = T.transform.translation, T.transform.rotation
                    ref_frame = Frame([t.x, t.y, t.z], quat.quaternion(r.w, r.x, r.y, r.z))
                
                
                # apply transformation/pose based on reference on:
                #   frame_1  if  INDIVIDUAL, COORDINATED, ABSOLUTE, RIGHT
                #   frame_2  if  INDIVIDUAL, LEFT (RELATIVE is always in ABSOLUTE frame)
                
                # next_posture = local_frame @ local_transformation
                if posture.reference == YumiPosture.REFERENCE_TOOL:
                    # provided `eX_t` compute bX_t = bHe @ eX_t
                    base_1, pose_1 = prev_1, frame_1
                    base_2, pose_2 = prev_2, frame_2
                
                elif posture.reference == YumiPosture.REFERENCE_LOCAL:
                    # provided `fX_t` compute bX_t = bHf @ fX_t
                    base_1, pose_1 = ref_frame, frame_1
                    base_2, pose_2 = ref_frame, frame_2
                
                # next_posture = frame_transformation @ prev_posture
                elif posture.reference == YumiPosture.REFERENCE_GLOBAL:
                    # provided `bHb'` compute bX_t = bHb' @ bX_i
                    base_1, pose_1 = frame_1, prev_1
                    base_2, pose_2 = frame_2, prev_2
                
                elif posture.reference == YumiPosture.REFERENCE_FRAME:
                    # provided `fHf'` compute bX_t = (bHf @ fHf' / bHf) @ bX_i
                    base_1, pose_1 = (ref_frame @ frame_1 / ref_frame), prev_1
                    base_2, pose_2 = (ref_frame @ frame_2 / ref_frame), prev_2
                           
                else:
                    error_string = f"Unknown reference type {posture.reference}"
                    rospy.logerr(error_string)
                    raise ArgumentError(error_string)
                
                frame_1 = base_1 @ pose_1
                if posture.motion in [YumiPosture.MOTION_INDIVIDUAL, YumiPosture.MOTION_LEFT]:
                    frame_2 = base_2 @ pose_2
            
            
            # TODO this must be fixed
            frame_1.vel = vel_1
            frame_2.vel = vel_2
            trajectory.append(((frame_1, frame_2, grip_r, grip_l), duration))
        
        transform_listener.unregister()
        
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
