import rospy, tf
import numpy as np, quaternion as quat

import threading
from collections import deque

from std_msgs.msg import Int64 as Int64Msg
from nav_msgs.msg import Path as PathMsg
from yumift_msgs.msg import YumiTrajectory as YumiTrajectoryMsg, YumiPosture as YumiPostureMsg

from ..common.controller_base import YumiDevice, YumiDualDeviceState, YumiDualDeviceAction
from ..common.controller_routinable import RoutinableYumiController
from ..common.control_laws import YumiDualCartesianVelocityControlLaw
from ..ik.algorithms import HQPIKAlgorithm, PINVIKAlgorithm
from ..misc.utils import (
    sanitize_pos, sanitize_rot, sanitize_vel, quat_to_xyzw,
    Frame_to_PoseStampedMsg, YumiParam_to_YumiCoordinatedRobotState
)

from .routines import ReadyPoseRoutine, CalibPoseRoutine
from .trajectory import YumiParam, YumiTrajectory, YumiTrajectoryParam


DEBUG = False

class YumiTrajectoryController(RoutinableYumiController):
    """ Class for running trajectory control using an instance of `YumiDualController`.
        Trajectory parameters are sent with ROS Message `YumiTrajectory` and from 
        those a `YumiTrajectory` is constructed and tracked using the chosen 
        `YumiDualCartesianVelocityControlLaw`.
    """
    # TODO control_law has wrong type
    def __init__(self, trajectory_topic: str, control_law: YumiDualCartesianVelocityControlLaw):
        super().__init__(
            robot_handle=YumiDevice(), 
            iksolvers=[PINVIKAlgorithm(), HQPIKAlgorithm()], 
            routines=[ReadyPoseRoutine(), CalibPoseRoutine()])
        self._iksolver.switch("pinv")
        
        # TODO , symmetry=0.
        
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
        rospy.Subscriber(trajectory_topic, YumiTrajectoryMsg, self._callback_trajectory, queue_size=1, tcp_nodelay=False)
        # TODO useful?
        self.pub_current_segment = rospy.Publisher("/trajectory_segment_progress", Int64Msg, queue_size=1, tcp_nodelay=False)
        
        # if DEBUG:
        ########################     VISUALIZATION     ########################
        # self._path_len_cache = 1
        # publish desired path
        # self._path_1 = deque(maxlen=self._path_len_cache)
        # self._path_2 = deque(maxlen=self._path_len_cache)
        self._pub_path_1 = rospy.Publisher("/path_1", PathMsg, tcp_nodelay=True, queue_size=1)
        self._pub_path_2 = rospy.Publisher("/path_2", PathMsg, tcp_nodelay=True, queue_size=1)
        # publish current and desired frames
        # self._broadcaster = tf.TransformBroadcaster()
        #######################################################################

    def reset(self, state: YumiDualDeviceState):
        """ Initialize the controller setting the current point as desired trajectory. 
            This method is called automatically every time EGM reconnects or after a 
            routine is completed.
        """
        # read current state of Yumi
        while True:
            if self._device_is_ready():
                current_pose = YumiParam(
                    state.pose_gripper_r.pos, state.pose_gripper_r.rot, np.zeros(6), 0, 
                    state.pose_gripper_l.pos, state.pose_gripper_l.rot, np.zeros(6), 0)
                break
            else:
                rospy.logerr("Controller cannot be reset (Yumi is not ready, retrying in 5 seconds)")
                rospy.sleep(5)
        # create the dummy path
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
        if DEBUG:
            self._path_1.clear()
            self._path_2.clear()
            self._pub_path_1.publish(PathMsg())
            self._pub_path_2.publish(PathMsg())
                
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
        curr_state = self._device_read()
        curr_pose_1, curr_pose_2 = curr_state.poses_individual if is_individual else curr_state.poses_coordinated
        grip_r, grip_l = curr_state.grip_r, curr_state.grip_l
        traj_point = YumiParam(curr_pose_1.pos, curr_pose_1.rot, curr_pose_1.vel, grip_r, 
                               curr_pose_2.pos, curr_pose_2.rot, curr_pose_2.vel, grip_l)
        # print(traj_point)
        trajectory = [YumiTrajectoryParam(traj_point, duration=0)]
        
        # append trajectory points from msg
        for posture in traj_msg.trajectory:
            # TODO use point.mode
            posture: YumiPostureMsg
            # right/absolute, left/relative, grippers
            if posture.incremental == YumiPostureMsg.OFF:
                # default values are previous ones
                prev_param = trajectory[-1].param
                pos_1 = sanitize_pos(posture.pose_primary.position,         prev_param.pose_right.pos)
                rot_1 = sanitize_rot(posture.pose_primary.orientation,      prev_param.pose_right.rot)
                vel_1 = sanitize_vel(posture.twist_primary,                 prev_param.pose_right.vel)
                pos_2 = sanitize_pos(posture.pose_secondary.position,       prev_param.pose_left.pos)
                rot_2 = sanitize_rot(posture.pose_secondary.orientation,    prev_param.pose_left.rot)
                vel_2 = sanitize_vel(posture.twist_secondary,               prev_param.pose_left.vel)
                grip_r = posture.gripper_right
                grip_l = posture.gripper_left
                
            else:
                # default values are zeros/unitary
                prev_param = trajectory[-1].param
                pos_1 = sanitize_pos(posture.pose_primary.position)
                rot_1 = sanitize_rot(posture.pose_primary.orientation)
                vel_1 = sanitize_vel(posture.twist_primary)
                pos_2 = sanitize_pos(posture.pose_secondary.position)
                rot_2 = sanitize_rot(posture.pose_secondary.orientation)
                vel_2 = sanitize_vel(posture.twist_secondary)
                grip_r = posture.gripper_right
                grip_l = posture.gripper_left
                
                pos_1, pos_2 = pos_1 + prev_param.pose_right.pos, pos_2 + prev_param.pose_left.pos
                grip_r, grip_l = grip_r + prev_param.grip_right, grip_l + prev_param.grip_left
                
                # TODO handle incremental twist
                
                # handle incremental postures
                if posture.incremental == YumiPostureMsg.LOCAL:
                    rot_1, rot_2 = rot_1 * prev_param.pose_right.rot, rot_2 * prev_param.pose_left.rot
                elif posture.incremental == YumiPostureMsg.GLOBAL:
                    rot_1, rot_2 = prev_param.pose_right.rot * rot_1, prev_param.pose_left.rot * rot_2
                else:
                    rospy.logerr(f"Unknown incremental mode {posture.incremental}")
                
            duration = posture.time_to_execute.to_sec()
            traj_point = YumiParam(pos_1, rot_1, vel_1, grip_r, pos_2, rot_2, vel_2, grip_l)
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
        
        msg = PathMsg()
        msg.header.frame_id = "yumi_base_link"
        msg.header.stamp = rospy.Time.now()
        msg.poses = [posture.pose_primary for posture in traj_msg.trajectory]
        print(msg.poses)
        self._pub_path_1.publish(msg)
        
        
    def policy(self, state: YumiDualDeviceState) -> YumiDualDeviceAction:
        """ Calculate target velocity for the current time step.
        """
        
        # START MODIFING THE TARGET
        self._lock_trajectory.acquire()

        # update timing information
        real_now = rospy.Time.now()
        state_now: rospy.Time = state.time
        dt = (real_now - state_now).to_sec()
        self.control_law.update_current_timestep(dt)
        
        # update pose and wrench for the control law class
        self.control_law.update_current_state(state)
        
        # calculate new desired velocities and positions for this time step
        yumi_desired_param : YumiParam = self.trajectory.compute((real_now - self.trajectory_initial_time).to_sec())
        yumi_desired_state = YumiParam_to_YumiCoordinatedRobotState(yumi_desired_param)
        
        self.control_law.update_desired_state(yumi_desired_state)
        
        if DEBUG:
            ########################     VISUALIZATION     ########################
            
            ### FRAMES
            
            # broadcast current coordinated poses
            self._broadcaster.sendTransform(state.pose_abs.pos, quat_to_xyzw(state.pose_abs.rot), rospy.Time.now(), "now_absolute_pose", "yumi_base_link")
            self._broadcaster.sendTransform(state.pose_rel.pos, quat_to_xyzw(state.pose_rel.rot), rospy.Time.now(), "now_relative_pose", "now_absolute_pose")
            # broadcast desired coordinated poses
            if not self.control_law.mode == self.control_law.ControlMode.INDIVIDUAL:
                self._broadcaster.sendTransform(yumi_desired_state.pose_gripper_r.pos, quat_to_xyzw(yumi_desired_state.pose_gripper_r.rot), rospy.Time.now(), "des_absolute_pose", "yumi_base_link")
                self._broadcaster.sendTransform(yumi_desired_state.pose_gripper_l.pos, quat_to_xyzw(yumi_desired_state.pose_gripper_l.rot), rospy.Time.now(), "des_relative_pose", "des_absolute_pose")
            
            ### PATHS
            
            # create desired pose for specified control mode
            des_parent_1, des_parent_2 = "yumi_base_link", "yumi_base_link" if self.control_law.mode == self.control_law.ControlMode.INDIVIDUAL else "des_absolute_pose"
            des_pose_1, des_pose_2 = yumi_desired_state.pose_gripper_r, yumi_desired_state.pose_gripper_l
            self._path_1.append(Frame_to_PoseStampedMsg(des_pose_1, des_parent_1))
            self._path_2.append(Frame_to_PoseStampedMsg(des_pose_2, des_parent_2))
            
            # publish everything
            path_1 = PathMsg()
            path_1.header.frame_id = des_parent_1
            path_1.header.stamp = rospy.Time.now()
            path_1.poses = list(self._path_1)
            self._pub_path_1.publish(path_1)
            
            path_2 = PathMsg()
            path_2.header.frame_id = des_parent_2
            path_2.header.stamp = rospy.Time.now()
            path_2.poses = list(self._path_2)
            self._pub_path_2.publish(path_2)
            #######################################################################
        
        # CALCULATE VELOCITIES
        try:
            # set velocities based on control mode
            vel_1, vel_2 = self.control_law.compute_target_state()
            
            # get space based on control mode ...
            action = YumiDualDeviceAction()
            action.control_space(YumiDualDeviceAction.ControlSpace.from_str(self.control_law.mode.value))
            action.timestep(dt)
            
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
            
        except Exception as ex:
            rospy.logfatal(f"Could not compute action (exception: {ex})")
            rospy.logfatal("Manually invoking fallback policy")
            action = self.fallback(state)
                
        # set commands to the grippers
        # (gripper commands should be sent only once per trajectory, the way they work is different)
        if self.trajectory.is_new_segment():
            action.gripper_right(yumi_desired_state.grip_r)
            action.gripper_left(yumi_desired_state.grip_l)
        
        # # sends information about which part of the trajectory is being executed
        # msg_segment = Int64Msg(data=self.trajectory.get_current_segment())
        # self.pub_current_segment.publish(msg_segment)
        
        self._lock_trajectory.release()
        
        return action
