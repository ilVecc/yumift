#!/usr/bin/env python3
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).absolute().parent.parent))
import argparse

from typing import Optional, List

import rospy, tf
import numpy as np
import quaternion as quat

import threading
from collections import deque

from std_msgs.msg import Int64
from yumi_controller.msg import Trajectory_msg, Trajectory_point
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

from core.controller import YumiDualController
from core.control_laws import YumiDualCartesianVelocityControlLaw, YumiDualWrenchFeedbackControlLaw, YumiDualAdmittanceControlLaw
from core.trajectory import YumiParam, YumiTrajectory, YumiTrajectoryParam
from core.parameters import Parameters
from core.utils import YumiParam_to_YumiCoordinatedRobotState

from gains import GAINS


class YumiTrajectoryController(YumiDualController):
    """ Class for running trajectory control using an instance of `YumiDualController`.
        Trajectory parameters are sent with ROS Message `Trajectory_msg` and from 
        those a `YumiTrajectory` is constructed and tracked using the chosen 
        `YumiDualCartesianVelocityControlLaw`.
    """
    def __init__(self, control_law: YumiDualCartesianVelocityControlLaw):
        super().__init__()
        self.yumi_state.alpha = 0.5
        
        # define control law
        self.control_law = control_law
        # The trajectory will always have two grippers, the control mode will 
        # always be either individual or coordinated, but the sent trajectory can 
        # also be only right/left or absolute/relative. To take this into account, the
        # trajectory and the target velocities will be calculated as a whole,
        # but the final selection of what will be sent to the inverse kinematics
        # solver is actually be performed based on what was originally requested.
        # Thus, `self.effective_mode` is used to store the original request.
        self.effective_mode = None
        
        # prepare trajectory buffer
        self.trajectory = YumiTrajectory(Parameters.dt)
        self.lock_trajectory = threading.Lock()
        self.reset()  # init trajectory (set current position)
        
        # listen for trajectory commands
        rospy.Subscriber("/trajectory", Trajectory_msg, self._callback_trajectory, tcp_nodelay=True, queue_size=1)
        self.pub_current_segment = rospy.Publisher("current_segment", Int64, tcp_nodelay=True, queue_size=1)
        
        ########################     VISUALIZATION     ########################
        self._path_len_cache = 500
        # publish desired path
        self._path_1 = deque(maxlen=self._path_len_cache)
        self._path_2 = deque(maxlen=self._path_len_cache)
        self._pub_path_1 = rospy.Publisher("path_1", Path, tcp_nodelay=True, queue_size=1)
        self._pub_path_2 = rospy.Publisher("path_2", Path, tcp_nodelay=True, queue_size=1)
        # publish current and desired frames
        self._broadcaster = tf.TransformBroadcaster()
        #######################################################################

    def reset(self):
        """ Initialize the controller setting the current point as desired trajectory. 
        """
        with self.lock_trajectory:
            self.control_law.clear()
            self.control_law.mode = "individual"
            self.effective_mode = self.control_law.mode
            current_pose = YumiParam(
                self.yumi_state.pose_gripper_r.pos, self.yumi_state.pose_gripper_r.rot, np.zeros(6), 0, 
                self.yumi_state.pose_gripper_l.pos, self.yumi_state.pose_gripper_l.rot, np.zeros(6), 0)
            path = [YumiTrajectoryParam(current_pose, 0), YumiTrajectoryParam(current_pose, 0.00001)]
            self.trajectory.update(path)
        print("Controller reset (previous trajectory has been discarded)")
    
    ########################     VISUALIZATION     ########################
    @staticmethod
    def _PoseParam_to_PoseStamped(pose, parent="yumi_base_link"):
        msg = PoseStamped()
        msg.header.frame_id = parent
        msg.header.stamp = rospy.Time.now()
        msg.pose.position.x = pose.pos[0]
        msg.pose.position.y = pose.pos[1]
        msg.pose.position.z = pose.pos[2]
        msg.pose.orientation.w = pose.rot.w
        msg.pose.orientation.x = pose.rot.x
        msg.pose.orientation.y = pose.rot.y
        msg.pose.orientation.z = pose.rot.z
        return msg
    ###########################################################################
    
    @staticmethod
    def _sanitize_pos(pos: Optional[List[float]]):
        return np.asarray(pos) if pos is not None else np.array([0,0,0])
    
    @staticmethod
    def _sanitize_rot(rot: Optional[List[float]]):
        return quat.from_float_array(np.roll(np.asarray(rot), 1)) if rot is not None else quat.one
    
    def _callback_trajectory(self, data: Trajectory_msg):
        """ Gets called when a new set of trajectory parameters is received. 
            The variable names in this function and the the trajectory class 
            follows individual motion with left and right. This means when 
            coordinate manipulation is used, right is absolute motion and left 
            becomes relative motion. 
        """
        self._path_1.clear()
        self._path_2.clear()
        self._pub_path_1.publish(Path())
        self._pub_path_2.publish(Path())
        
        # go through allowed routines (eg. "routine_reset_pose")
        if data.mode.startswith("routine_"):
            routine_name = data.mode[8:]  # (eg. "reset_pose")
            self.request_routine(routine_name)
            return
        
        if data.mode not in ["individual", "right", "left", "coordinated", "absolute", "relative"]:
            print(f"Error, mode \"{data.mode}\" is unknown.")
            return
        is_individual = data.mode in ["individual", "right", "left"]
        
        ########################   PREPARE TRAJECTORY   #######################
        # use current position, rotation and velocity as first trajectory points
        curr_pose_1, curr_pose_2 = self.yumi_state.poses_individual if is_individual else self.yumi_state.poses_coordinated
        grip_r = self.control_law.grip_r
        grip_l = self.control_law.grip_l
        currentPoint = YumiParam(curr_pose_1.pos, curr_pose_1.rot, curr_pose_1.vel, grip_r, 
                                 curr_pose_2.pos, curr_pose_2.rot, curr_pose_2.vel, grip_l)
        trajectory = [YumiTrajectoryParam(currentPoint, duration=0)]
        
        # append trajectory points from msg
        # ATTENTION: Trajectory_point wants quaternions as (x,y,z,w) while all 
        #            quaternions in the code are as (w,x,y,z)
        # TODO fix this
        for point in data.trajectory:
            point: Trajectory_point
            # either right or absolute
            pos_1 = self._sanitize_pos(point.positionRight if is_individual else point.positionAbsolute)
            rot_1 = self._sanitize_rot(point.orientationRight if is_individual else point.orientationAbsolute)
            # either left or relative
            pos_2 = self._sanitize_pos(point.positionLeft if is_individual else point.positionRelative)
            rot_2 = self._sanitize_rot(point.orientationLeft if is_individual else point.orientationRelative)
            # set the grippers
            grip_r = point.gripperRight
            grip_l = point.gripperLeft
            # if coordinates are relative
            if point.local:
                prev_param = trajectory[-1].param
                pos_1, rot_1 = pos_1 + prev_param.pose_right.pos, rot_1 * prev_param.pose_right.rot
                pos_2, rot_2 = pos_2 + prev_param.pose_left.pos,  rot_2 * prev_param.pose_left.rot
                grip_r, grip_l = grip_r + prev_param.grip_right, grip_l + prev_param.grip_left
            duration = point.pointTime
            trajectoryPoint = YumiParam(pos_1, rot_1, None, grip_r, pos_2, rot_2, None, grip_l)
            trajectory.append(YumiTrajectoryParam(trajectoryPoint, duration))
        #######################################################################
        
        # update the trajectory
        with self.lock_trajectory:
            self.control_law.mode = "individual" if is_individual else "coordinated"
            self.effective_mode = data.mode
            self.trajectory.update(trajectory)
        print(f"New trajectory received in \"{self.effective_mode}\" mode")
    
    @staticmethod
    def _wxyz_to_xyzw(q: np.quaternion):
        return np.roll(quat.as_float_array(q), -1)
    
    def policy(self):
        """ Calculate target velocity for the current time step.
        """
        # START MODIFING THE TARGET
        self.lock_trajectory.acquire()
        
        # update pose and wrench for the control law class
        self.control_law.update_current_pose(self.yumi_state)
        
        # calculate new target velocities and positions for this time step
        yumi_target_param: YumiParam = self.trajectory.compute()
        yumi_target_state = YumiParam_to_YumiCoordinatedRobotState(yumi_target_param)
        self.control_law.update_desired_pose(yumi_target_state)

        ########################     VISUALIZATION     ########################
        # broadcast current coordinated poses
        self._broadcaster.sendTransform(self.yumi_state.pose_abs.pos, self._wxyz_to_xyzw(self.yumi_state.pose_abs.rot), rospy.Time.now(), "now_absolute_pose", "yumi_base_link")
        self._broadcaster.sendTransform(self.yumi_state.pose_rel.pos, self._wxyz_to_xyzw(self.yumi_state.pose_rel.rot), rospy.Time.now(), "now_relative_pose", "now_absolute_pose")
        # broadcast desired coordinated poses
        if not self.control_law.mode == "individual":
            self._broadcaster.sendTransform(yumi_target_state.pose_gripper_r.pos, self._wxyz_to_xyzw(yumi_target_state.pose_gripper_r.rot), rospy.Time.now(), "des_absolute_pose", "yumi_base_link")
            self._broadcaster.sendTransform(yumi_target_state.pose_gripper_l.pos, self._wxyz_to_xyzw(yumi_target_state.pose_gripper_l.rot), rospy.Time.now(), "des_relative_pose", "des_absolute_pose")
        # create desired pose for specified control mode
        des_parent_1, des_parent_2 = "yumi_base_link", "yumi_base_link" if self.control_law.mode == "individual" else "des_absolute_pose"
        des_pose_1, des_pose_2 = yumi_target_state.pose_gripper_r, yumi_target_state.pose_gripper_l
        self._path_1.append(self._PoseParam_to_PoseStamped(des_pose_1, des_parent_1))
        self._path_2.append(self._PoseParam_to_PoseStamped(des_pose_2, des_parent_2))
        
        # publish everything
        path_1 = Path()
        path_1.header.frame_id = des_parent_1
        path_1.header.stamp = rospy.Time.now()
        path_1.poses = list(self._path_1)
        
        path_2 = Path()
        path_2.header.frame_id = des_parent_2
        path_2.header.stamp = rospy.Time.now()
        path_2.poses = list(self._path_2)
        
        self._pub_path_1.publish(path_1)
        self._pub_path_2.publish(path_2)
        #######################################################################
        
        # CALCULATE VELOCITIES
        action = dict()
        # set velocities based on control mode
        try:
            # get space based on control mode ...
            action["control_space"] = self.control_law.mode
            vel_1, vel_2 = self.control_law.compute_target_velocity()
            # ... but use the effective mode to set the velocities
            if self.effective_mode == "individual":
                action["right_velocity"], action["left_velocity"] = vel_1, vel_2
            elif self.effective_mode == "right":
                action["right_velocity"] = vel_1
            elif self.effective_mode == "left":
                action["left_velocity"] = vel_2
            elif self.effective_mode == "coordinated":
                action["absolute_velocity"], action["relative_velocity"] = vel_1, vel_2
            elif self.effective_mode == "absolute":
                action["absolute_velocity"] = vel_1
            elif self.effective_mode == "relative":
                action["relative_velocity"] = vel_2
                
        except Exception as ex:
            print(f"Stopping motion (exception: {ex})")
            action = {
                "control_space": "joint_space",
                "joint_velocities": np.zeros(Parameters.dof)}
                
        # set commands to the grippers
        # (gripper commands should be sent only once per trajectory, the way they work is different)
        if self.trajectory.is_new_segment():
            action["gripper_right"] = self.control_law.grip_r
            action["gripper_left"] = self.control_law.grip_l
        
        # sends information about which part of the trajectory is being executed
        msg_segment = Int64(data=self.trajectory.get_current_segment())
        self.pub_current_segment.publish(msg_segment)
        
        self.lock_trajectory.release()
        
        return action


class SimpleTrajectoryController(YumiTrajectoryController):
    def __init__(self):
        super().__init__(YumiDualCartesianVelocityControlLaw(GAINS))
        
class WrenchedTrajectoryController(YumiTrajectoryController):
    def __init__(self):
        super().__init__(YumiDualWrenchFeedbackControlLaw(GAINS))

class CompliantTrajectoryController(YumiTrajectoryController):
    def __init__(self):
        super().__init__(YumiDualAdmittanceControlLaw(GAINS))


def main():
    
    parser = argparse.ArgumentParser("Various trajectory controllers")
    parser.add_argument("type", nargs="?", choices=["simple", "wrenched", "compliant"], default="simple", type=str)
    args = parser.parse_args()
    
    # starting ROS node
    rospy.init_node("trajectory_controllers", anonymous=False)
    
    if args.type == "simple":
        yumi_controller = SimpleTrajectoryController()
    elif args.type == "wrenched":
        yumi_controller = WrenchedTrajectoryController()
    elif args.type == "compliant":
        yumi_controller = CompliantTrajectoryController()
    else:
        raise AttributeError(f"no such option '{parser.type}'")
    
    def shutdown_callback():
        yumi_controller.pause()
        print("Controller shutting down")
        rospy.sleep(0.5)
    
    rospy.on_shutdown(shutdown_callback)
    
    yumi_controller.start()


if __name__ == "__main__":
    main()