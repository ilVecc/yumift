#!/usr/bin/env python3
from typing import Tuple

import rospy, tf
import numpy as np
import quaternion as quat

import threading

from yumift_msgs.msg import (
    YumiTrajectory as YumiTrajectoryMsg, 
    YumiPosture as YumiPostureMsg
)
import yumift_common.msg_utils as msg_utils

from yumift_controllers.common.device import YumiDevice, YumiDualDeviceState
from yumift_controllers.common.controller_base import MixedVelocityYumiAction
from yumift_controllers.common.controller_routinable import RoutinableYumiController
from yumift_controllers.common.control_laws import YumiIndividualCartesianVelocityControlLaw
from yumift_controllers.impl.routines import ReadyPoseRoutine, CalibPoseRoutine
from yumift_controllers.impl.trajectory import YumiParam, YumiTrajectory, YumiTrajectoryParam
from yumift_controllers.misc.utils import load_config


class KentaurTrajectoryController(RoutinableYumiController):
    """ Class for running trajectory control using an instance of `YumiDualController`.
        Trajectory parameters are sent with ROS Message `YumiTrajectory` and from 
        those a `YumiTrajectory` is constructed and tracked using the chosen 
        `YumiDualCartesianVelocityControlLaw`.
    """
    def __init__(self):
        super().__init__(
            robot_handle=YumiDevice(), 
            iksolvers="pinv", 
            routines=[ReadyPoseRoutine(), CalibPoseRoutine()])
        
        trajectory_topic = "/trajectory"
        
        # define control law
        self.control_law = YumiIndividualCartesianVelocityControlLaw(load_config("gains.yaml")["CLIK"])
        # The trajectory will always have two grippers, the control mode will 
        # always be either individual or coordinated, but the sent trajectory can 
        # also be only right/left or absolute/relative. To take this into account, the
        # trajectory and the target velocities will be calculated as a whole,
        # but the final selection of what will be sent to the inverse kinematics
        # solver is actually be performed based on what was originally requested.
        # Thus, `self.effective_mode` is used to store the original request.
        self.effective_mode = None
        
        # prepare trajectory buffer
        self.trajectory_initial_time = rospy.Time.now()
        self.trajectory = YumiTrajectory()
        self._lock_trajectory = threading.Lock()
        
        # listen for trajectory commands
        self.current_state : YumiDualDeviceState = None
        rospy.Subscriber(trajectory_topic, YumiTrajectoryMsg, self._callback_trajectory, queue_size=1, tcp_nodelay=False)
        
    def reset(self, state: YumiDualDeviceState):
        """ Initialize the controller setting the current point as desired trajectory. 
            This method is called automatically every time EGM reconnects or after a 
            routine is completed.
        """
        # read current state of Yumi
        while True:
            self.device_read()
            if self.device_is_ready():
                current_pose = YumiParam(
                    state.pose_gripper_r.pos, state.pose_gripper_r.rot, np.zeros(6), 0, 
                    state.pose_gripper_l.pos, state.pose_gripper_l.rot, np.zeros(6), 0)
                break
            else:
                print("Controller cannot be reset (Yumi is not ready, retrying in 5 seconds)")
                rospy.sleep(5)
        # create the dummy path
        path = [YumiTrajectoryParam(current_pose, 0), YumiTrajectoryParam(current_pose, 0.00001)]
        # update the trajectory
        with self._lock_trajectory:
            self.control_law.clear()
            self.effective_mode = self.control_law.mode
            self.trajectory.update(path)
            self.trajectory_initial_time = rospy.Time.now()
        print("Controller reset (previous trajectory has been discarded)")
            
    @staticmethod
    def _sanitize_pos(pos: Tuple[float]):
        return np.asarray(pos) if pos else np.array([0,0,0])
    
    @staticmethod
    def _sanitize_rot(rot: Tuple[float]):
        return quat.quaternion(*rot) if rot else quat.one
    
    def _callback_trajectory(self, data: YumiTrajectoryMsg):
        """ Gets called when a new set of trajectory parameters is received. 
            The variable names in this function and the the trajectory class 
            follows individual motion with left and right. This means when 
            coordinate manipulation is used, right is absolute motion and left 
            becomes relative motion. 
        """
        # go through allowed routines (eg. "routine_ready_pose")
        if data.mode.startswith("routine_"):
            routine_name = data.mode[8:]  # (eg. "ready_pose")
            self.request_routine(routine_name)
            return
        
        if data.mode not in ["individual", "right", "left", "coordinated", "absolute", "relative"]:
            print(f"Error, mode \"{data.mode}\" is unknown.")
            return
        is_individual = data.mode in ["individual", "right", "left"]
        
        ########################   PREPARE TRAJECTORY   #######################
        # use current position, rotation and velocity as first trajectory points
        curr_pose_1, curr_pose_2 = self.current_state.poses_individual if is_individual else self.current_state.poses_coordinated
        grip_r, grip_l = self.current_state.grip_r, self.current_state.grip_l
        currentPoint = YumiParam(curr_pose_1.pos, curr_pose_1.rot, curr_pose_1.vel, grip_r, 
                                 curr_pose_2.pos, curr_pose_2.rot, curr_pose_2.vel, grip_l)
        trajectory = [YumiTrajectoryParam(currentPoint, duration=0)]
        
        # append trajectory points from msg
        for point in data.trajectory:
            point: YumiPostureMsg
            # either right or absolute
            pos_1 = self._sanitize_pos(point.positionRight if is_individual else point.positionAbsolute)
            rot_1 = self._sanitize_rot(point.orientationRight if is_individual else point.orientationAbsolute)
            # either left or relative
            pos_2 = self._sanitize_pos(point.positionLeft if is_individual else point.positionRelative)
            rot_2 = self._sanitize_rot(point.orientationLeft if is_individual else point.orientationRelative)
            # set the grippers
            grip_r = point.gripper_right
            grip_l = point.gripper_left
            # if coordinates are relative
            if point.incremental != YumiPostureMsg.OFF:
                prev_param = trajectory[-1].param
                pos_1, rot_1 = pos_1 + prev_param.pose_right.pos, rot_1 * prev_param.pose_right.rot
                pos_2, rot_2 = pos_2 + prev_param.pose_left.pos,  rot_2 * prev_param.pose_left.rot
                grip_r, grip_l = grip_r + prev_param.grip_right, grip_l + prev_param.grip_left
            duration = point.time_to_execute
            trajectory_point = YumiParam(pos_1, rot_1, None, grip_r, pos_2, rot_2, None, grip_l)
            trajectory.append(YumiTrajectoryParam(trajectory_point, duration))
        #######################################################################
        
        # update the trajectory
        with self._lock_trajectory:
            self.control_law.mode = "individual" if is_individual else "coordinated"
            self.effective_mode = data.mode
            self.trajectory.update(trajectory)
            self.trajectory_initial_time = rospy.Time.now()
        print(f"New trajectory received in \"{self.effective_mode}\" mode")
    
    def policy(self, state: YumiDualDeviceState) -> MixedVelocityYumiAction:
        """ Calculate target velocity for the current time step.
        """
        
        # need to give the state to the trajectory callback
        self.current_state = state
        
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
        yumi_desired_param: YumiParam = self.trajectory.compute((real_now - self.trajectory_initial_time).to_sec())
        yumi_desired_state = msg_utils.YumiParam_to_YumiCoordinatedRobotState(yumi_desired_param)
        
        self.control_law.update_desired_state(yumi_desired_state)
        
        # CALCULATE VELOCITIES
        action = MixedVelocityYumiAction()
        
        # set velocities based on control mode
        try:
            # get space based on control mode ...
            action.control_space(MixedVelocityYumiAction.ControlSpace.from_str(self.control_law.mode))
            action.timestep(dt)
            
            vel_1, vel_2 = self.control_law.compute_target_state()
            
            # ... but use the effective mode to set the velocities
            if self.effective_mode == "individual":
                action.velocity_right(vel_1) 
                action.velocity_left(vel_2)
            elif self.effective_mode == "right":
                action.velocity_right(vel_1)
            elif self.effective_mode == "left":
                action.velocity_left(vel_2)
            elif self.effective_mode == "coordinated":
                action.velocity_absolute(vel_1)
                action.velocity_relative(vel_2)
            elif self.effective_mode == "absolute":
                action.velocity_absolute(vel_1)
            elif self.effective_mode == "relative":
                action.velocity_relative(vel_2)
            
        except Exception as ex:
            print(f"Stopping motion (exception: {ex})")
            action = self.fallback(state)
                
        # set commands to the grippers
        # (gripper commands should be sent only once per trajectory, the way they work is different)
        if self.trajectory.is_new_segment():
            action.gripper_right(yumi_desired_state.grip_r)
            action.gripper_left(yumi_desired_state.grip_l)
        
        self._lock_trajectory.release()
        
        return action


def main():
    
    # starting ROS node
    rospy.init_node("trajectory_controllers", anonymous=False)
    
    kentaur_controller = KentaurTrajectoryController()
    
    def shutdown_callback():
        print("Controller shutting down")
        kentaur_controller.stop()
    
    rospy.on_shutdown(shutdown_callback)
    
    kentaur_controller.ready()
    kentaur_controller.start()  # locking


if __name__ == "__main__":
    main()
