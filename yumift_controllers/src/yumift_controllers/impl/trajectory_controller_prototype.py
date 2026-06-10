from typing import List
from typing_extensions import override

import rospy, tf
import numpy as np, quaternion as quat

import threading

from std_msgs.msg import Int64 as Int64Msg
from nav_msgs.msg import Path as PathMsg
from geometry_msgs.msg import Point as PointMsg
from visualization_msgs.msg import MarkerArray as MarkerArrayMsg, Marker as MarkerMsg
from yumift_msgs.msg import YumiTrajectory as YumiTrajectoryMsg, YumiPosture as YumiPostureMsg

from ..common.device import YumiDevice, YumiDualDeviceState, YumiCoordinatedRobotState
from ..common.controller_base import MixedVelocityYumiAction
from ..common.controller_routinable import RoutinableYumiController
from ..common.control_laws import YumiDualCartesianVelocityControlLaw
from ..ik.algorithms import HQPIKAlgorithm, PINVIKAlgorithm
from ..misc.utils import (
    sanitize_pos, sanitize_rot, sanitize_vel, sanitize_grip, quat_to_xyzw,
    Frame_to_PoseStampedMsg, YumiCoordinatedRobotState_from_YumiParam
)
from dynamicals.utils import Frame

from .routines import ReadyPoseRoutine, CalibPoseRoutine
from .trajectory import YumiParam, YumiTrajectory, YumiTrajectoryParam, PoseParam


class TrajectoryVisualizers():
    
    def __init__(self, ns):
        self._is_individual = False
        self._path_1 = []
        self._path_2 = []
        self._pub_now_path_1 = rospy.Publisher(ns+"/visualization/now_path_1", PathMsg, queue_size=1)
        self._pub_now_path_2 = rospy.Publisher(ns+"/visualization/now_path_2", PathMsg, queue_size=1)
        self._pub_des_path_1 = rospy.Publisher(ns+"/visualization/des_path_1", PathMsg, queue_size=1)
        self._pub_des_path_2 = rospy.Publisher(ns+"/visualization/des_path_2", PathMsg, queue_size=1)
        self._pub_viz = rospy.Publisher(ns+"/visualization/markers", MarkerArrayMsg, queue_size=1)
        self._broadcaster = tf.TransformBroadcaster()
    
    @staticmethod
    def _create_path_message(frame: str, id: int, points: List[PointMsg]) -> MarkerMsg:
        msg = MarkerMsg(ns="paths", id=id, type=MarkerMsg.LINE_STRIP, action=MarkerMsg.MODIFY)
        msg.header.frame_id = frame
        msg.header.stamp = rospy.Time.now()
        msg.pose.orientation.w = 1.0
        msg.scale.x = 0.001
        msg.color.a = 1.0
        msg.color.r = 1.0
        msg.color.g = 85/255
        msg.color.b = 0.0
        msg.points = points
        return msg
    
    def prepare(self, targets: List[YumiParam], is_individual: bool):
        self._is_individual = is_individual
        
        msg = PathMsg()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "yumi_base_link"
        # msg.poses = [PoseStampedMsg(msg.header, posture.pose_primary) for posture in traj_msg.trajectory]
        msg.poses = [Frame_to_PoseStampedMsg(PoseParam.to_Frame(target.pose_right)) for target in targets]
        self._pub_des_path_1.publish(msg)
        
        msg.header.frame_id = "yumi_base_link" if is_individual else "des_abs"
        # msg.poses = [PoseStampedMsg(msg.header, posture.pose_secondary) for posture in traj_msg.trajectory]
        msg.poses = [Frame_to_PoseStampedMsg(PoseParam.to_Frame(target.pose_left)) for target in targets]
        self._pub_des_path_2.publish(msg)
        
        self._path_1 = []
        self._path_2 = []
        
        self._pub_viz.publish(MarkerArrayMsg(markers=[
            self._create_path_message("yumi_base_link", id=0, points=[
                PointMsg(target.pose_right.pos[0], target.pose_right.pos[1], target.pose_right.pos[2]) for target in targets]),
            self._create_path_message("yumi_base_link", id=1, points=[
                PointMsg(target.pose_left.pos[0], target.pose_left.pos[1], target.pose_left.pos[2]) for target in targets])
        ]))
    
    def publish(self, state: YumiDualDeviceState, yumi_desired_state: YumiCoordinatedRobotState):
        now = rospy.Time.now()
        # broadcast current coordinated poses
        self._broadcaster.sendTransform(state.pose_abs.pos, quat_to_xyzw(state.pose_abs.rot), now, "now_abs", "yumi_base_link")
        self._broadcaster.sendTransform(state.pose_rel.pos, quat_to_xyzw(state.pose_rel.rot), now, "now_rel", "now_abs")
        
        # create desired pose for specified control mode
        des_parent_1, des_parent_2 = "yumi_base_link", ("yumi_base_link" if self._is_individual else "des_abs")
        des_pose_1, des_pose_2 = yumi_desired_state.pose_gripper_r, yumi_desired_state.pose_gripper_l
        self._path_1.append(Frame_to_PoseStampedMsg(des_pose_1, des_parent_1))
        self._path_2.append(Frame_to_PoseStampedMsg(des_pose_2, des_parent_2))
        
        # publish everything
        msg = PathMsg()
        msg.header.stamp = now
        msg.header.frame_id = des_parent_1
        msg.poses = self._path_1
        self._pub_now_path_1.publish(msg)
        msg.header.frame_id = des_parent_2
        msg.poses = self._path_2
        self._pub_now_path_2.publish(msg)


class YumiTrajectoryController(RoutinableYumiController):
    """ Class for running trajectory control using an instance of `YumiDualController`.
        Trajectory parameters are sent with ROS Message `YumiTrajectory` and from 
        those a `YumiTrajectory` is constructed and tracked using the chosen 
        `YumiDualCartesianVelocityControlLaw`.
    """
    # TODO control_law has wrong type
    def __init__(self, trajectory_topic: str, control_law: YumiDualCartesianVelocityControlLaw, debug: bool = False):
        super().__init__(
            robot_handle=YumiDevice(coordinated_balance=0.5), 
            iksolvers=[PINVIKAlgorithm(), HQPIKAlgorithm()], 
            routines=[ReadyPoseRoutine(), CalibPoseRoutine()])
        self._iksolver.switch("pinv")
        
        # define control law
        self.control_law = control_law
        # The trajectory will always have two grippers, the control mode will 
        # always be either individual or coordinated, but the sent trajectory can 
        # also be only right/left or absolute/relative. To take this into account, the
        # trajectory and the target velocities will be calculated as a whole,
        # but the final selection of what will be sent to the inverse kinematics
        # solver is actually be performed based on what was originally requested.
        # Thus, `self.effective_mode` is used to store the original request.
        self.effective_mode : int = None
        
        # prepare trajectory buffer
        self.trajectory_initial_time = rospy.Time.now()
        self.trajectory = YumiTrajectory()
        self._lock_trajectory = threading.Lock()
        
        # listen for trajectory commands
        rospy.Subscriber(trajectory_topic, YumiTrajectoryMsg, self._callback_trajectory, queue_size=1)
        # TODO useful?
        self.pub_current_segment = rospy.Publisher(trajectory_topic+"/segment_progress", Int64Msg, queue_size=1)
        
        self.debug = debug
        if self.debug:
            self.viz = TrajectoryVisualizers(trajectory_topic)

    @override
    def reset(self, state: YumiDualDeviceState):
        """ Initialize the controller setting the current point as desired trajectory. 
            This method is called automatically every time EGM reconnects or after a 
            routine is completed.
        """
        # wait for Yumi
        while not self.device_is_ready():
            rospy.logwarn("Controller cannot be reset (Yumi is not ready, retrying in 5 seconds)")
            rospy.sleep(5)
        # create the dummy path
        current_pose = YumiParam.from_Frames(state.pose_gripper_r.motionless(), state.pose_gripper_l.motionless(), 0, 0)
        path = [YumiTrajectoryParam(current_pose, 0), YumiTrajectoryParam(current_pose, 0.00001)]
        # update the trajectory
        with self._lock_trajectory:
            self.control_law.clear()
            self.effective_mode = YumiTrajectoryMsg.INDIVIDUAL
            self.trajectory.update(path)
            self.trajectory_initial_time = rospy.Time.now()
        rospy.loginfo("Controller reset (previous trajectory has been discarded)")
    
    def _callback_trajectory(self, traj_msg: YumiTrajectoryMsg):
        """ Gets called when a new set of trajectory parameters is received. 
            The variable names in this function and the the trajectory class 
            follows individual motion with left and right. This means when 
            coordinate manipulation is used, right is absolute motion and left 
            becomes relative motion. 
        """
        if traj_msg.mode == YumiTrajectoryMsg.ROUTINE:
            if traj_msg.routine_name == "":
                rospy.logwarn("No routine specified, resuming")
                return
            self.request_routine(traj_msg.routine_name)  # (eg. "ready_pose")
            return
            
        if traj_msg.mode not in [
            YumiTrajectoryMsg.INDIVIDUAL, YumiTrajectoryMsg.RIGHT, YumiTrajectoryMsg.LEFT, 
            YumiTrajectoryMsg.COORDINATED, YumiTrajectoryMsg.ABSOLUTE, YumiTrajectoryMsg.RELATIVE]:
            rospy.logwarn(f"Control mode \"{traj_msg.mode}\" is invalid, resuming")
            return
        
        is_individual = traj_msg.mode in [
            YumiTrajectoryMsg.INDIVIDUAL, YumiTrajectoryMsg.RIGHT, YumiTrajectoryMsg.LEFT]
        
        ########################   PREPARE TRAJECTORY   #######################
        # use current position, rotation and velocity as first trajectory points
        # use the required mode as first pose
        curr_state = self.device_read()
        curr_pose_1, curr_pose_2 = curr_state.poses_individual if is_individual else curr_state.poses_coordinated
        traj_point = YumiParam.from_Frames(curr_pose_1, curr_pose_2, curr_state.grip_r, curr_state.grip_l)
        trajectory = [YumiTrajectoryParam(traj_point, duration=0)]
        
        # append trajectory points from msg
        for posture in traj_msg.trajectory:
            posture: YumiPostureMsg
            pos_1 = sanitize_pos(posture.pose_primary.position, default_none=False)
            rot_1 = sanitize_rot(posture.pose_primary.orientation, default_none=False)
            vel_1 = sanitize_vel(posture.twist_primary)
            frame_1 = Frame(pos_1, rot_1) #, vel_1)
            pos_2 = sanitize_pos(posture.pose_secondary.position, default_none=False)
            rot_2 = sanitize_rot(posture.pose_secondary.orientation, default_none=False)
            vel_2 = sanitize_vel(posture.twist_secondary)
            frame_2 = Frame(pos_2, rot_2) #, vel_2)
            grip_r = sanitize_grip(posture.gripper_right, default_none=False)
            grip_l = sanitize_grip(posture.gripper_left, default_none=False)
            duration = posture.time_to_execute.to_sec()
            # posture.mode  # TODO use me
            
            # convert everything to GLOBAL COORDINATES
            if posture.incremental != YumiPostureMsg.OFF:
                prev_param = trajectory[-1].param
                prev_1 = PoseParam.to_Frame(prev_param.pose_right)
                prev_2 = PoseParam.to_Frame(prev_param.pose_left)
                grip_r = grip_r + prev_param.grip_right
                grip_l = grip_l + prev_param.grip_left
                
                # TODO handle twist (can be None) in incremental mode
                prev_1.vel = np.zeros(6)
                prev_2.vel = np.zeros(6)
                
                # handle incremental postures
                if posture.incremental == YumiPostureMsg.LOCAL:
                    # next_posture = prev_posture @ local_transformation
                    frame_1 = prev_1 @ frame_1
                    frame_2 = prev_2 @ frame_2
                elif posture.incremental == YumiPostureMsg.GLOBAL:
                    # next_posture = global_transformation @ prev_posture
                    frame_1 = frame_1 @ prev_1
                    frame_2 = frame_2 @ prev_2
                else:
                    rospy.logerr(f"Unknown incremental mode {posture.incremental}")
            
            traj_point = YumiParam(frame_1.pos, frame_1.rot, vel_1, grip_r, frame_2.pos, frame_2.rot, vel_2, grip_l)
            trajectory.append(YumiTrajectoryParam(traj_point, duration))
        #######################################################################
        
        # update the trajectory
        CM = self.control_law.ControlMode
        with self._lock_trajectory:
            self.control_law.mode = CM.INDIVIDUAL if is_individual else CM.COORDINATED
            self.effective_mode = traj_msg.mode
            self.trajectory.update(trajectory)
            self.trajectory_initial_time = rospy.Time.now()
        rospy.loginfo(f"New trajectory received in \"{self.control_law.mode.name}\" mode")
        
        if self.debug:    
            time = sum([point.duration for point in trajectory])
            targets = [self.trajectory.compute(t) for t in np.linspace(0, time, 50, endpoint=True)]
            self.trajectory.restart()
            self.viz.prepare(targets, is_individual)
    
    @override
    def policy(self, state: YumiDualDeviceState) -> MixedVelocityYumiAction:
        """ Calculate target velocity for the current time step.
        """
        # TODO can we shorten the usage of this lock?
        self._lock_trajectory.acquire()
        
        # calculate timing information
        state_dt = self.dt(state.time)
        traj_dt = self.dt(self.trajectory_initial_time)
        
        # update timing
        self.control_law.update_current_timestep(state_dt)
        
        # update pose and wrench for the control law class
        self.control_law.update_current_state(state)
        
        # calculate new desired velocities and positions for this time step
        des_param : YumiParam = self.trajectory.compute(traj_dt)
        des_state = YumiCoordinatedRobotState_from_YumiParam(des_param)
        self.control_law.update_desired_state(des_state)
        
        # CALCULATE VELOCITIES
        try:
            # set velocities based on control mode
            vel_1, vel_2 = self.control_law.compute_target_state()
            # print()
            
            # get space based on control mode ...
            action = MixedVelocityYumiAction()
            action.control_space(MixedVelocityYumiAction.ControlSpace.from_str(self.control_law.mode.value))
            action.timestep(state_dt)
            
            # ... but use the effective mode to set the velocities
            if self.effective_mode == YumiTrajectoryMsg.INDIVIDUAL:
                action.velocity_right(vel_1) 
                action.velocity_left(vel_2)
            elif self.effective_mode == YumiTrajectoryMsg.RIGHT:
                action.velocity_right(vel_1)
            elif self.effective_mode == YumiTrajectoryMsg.LEFT:
                action.velocity_left(vel_2)
            elif self.effective_mode == YumiTrajectoryMsg.COORDINATED:
                action.velocity_absolute(vel_1)
                action.velocity_relative(vel_2)
            elif self.effective_mode == YumiTrajectoryMsg.ABSOLUTE:
                action.velocity_absolute(vel_1)
            elif self.effective_mode == YumiTrajectoryMsg.RELATIVE:
                action.velocity_relative(vel_2)
            
            # set commands to the grippers
            # (gripper commands should be sent only once per trajectory, the 
            # way they work is different)
            is_new_traj_segment = self.trajectory.is_new_segment()
            if is_new_traj_segment:
                action.gripper_right(des_state.grip_r)
                action.gripper_left(des_state.grip_l)
            
        except Exception as ex:
            rospy.logfatal(f"Could not compute action (exception: {ex})")
            rospy.logfatal("Manually invoking fallback policy")
            action = self.fallback(state)
        
        # sends information about which part of the trajectory is being executed
        self.pub_current_segment.publish(Int64Msg(self.trajectory.get_current_segment()))
        
        self._lock_trajectory.release()
        
        if self.debug:
            self.viz.publish(state, des_state)
        
        return action
