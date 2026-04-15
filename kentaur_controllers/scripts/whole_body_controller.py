#!/usr/bin/env python3
from typing_extensions import override

import rospy
import numpy as np, quaternion as quat

from dynamicals.utils import Frame, jacobian_change_base_frame
from dynamicals.common.controllers import AbstractController

from pathfinder import PoseParam, CubicPoseTrajectory

from yumift_controllers.common.control_laws import YumiDualAdmittanceControlLaw
from yumift_controllers.impl.trajectory import YumiParam
from yumift_controllers.misc.utils import YumiParam_to_YumiCoordinatedRobotState

from yumift_common.constants import YumiRobotConstants

from kentaur_controllers.common.device import (
    KentaurDeviceState, KentaurDeviceCommand, KentaurDevice,
    YumiDualDeviceCommand, SleipnerCartesianDeviceCommand
)
from kentaur_controllers.common.controller_base import KentaurDeviceAction, MixedVelocityYumiAction
from kentaur_controllers.misc.utils import load_config, PoseParam_to_Frame


class KentaurController(AbstractController[KentaurDeviceState, KentaurDeviceAction, KentaurDeviceCommand]):

    def __init__(self, kentaur_device : KentaurDevice):
        self._device : KentaurDevice
        super().__init__(kentaur_device)
        self.control_law = YumiDualAdmittanceControlLaw(load_config("gains_whole_body.yaml"))
        self.trajectory_r = CubicPoseTrajectory()
        self.trajectory_l = CubicPoseTrajectory()
        self.trajectory_initial_time = rospy.Time.now()
        
        # reset gripper position difference, in home frame
        # self.diff_r = Frame(np.array([-0.2, -0.1, 0]))
        # self.diff_l = Frame(np.array([-0.2, +0.1, 0]))
        self.diff_r = Frame(np.array([0, 0, 0]))
        self.diff_l = Frame(np.array([0, 0, 0]))
    
    @override
    def start(self):
        self.reset(self.device_read())  # init trajectory
        super().start(250)

    @override
    def stop(self):
        rospy.loginfo("Controller shutting down")
        return super().stop()

    @override
    def spin(self, control_rate: float):
        """ ROS implementation of the original function.
        """
        rate = rospy.Rate(control_rate)
        while not rospy.is_shutdown():
            self.spin_once()
            rate.sleep()
        # when the controller is shut down, send a stop command
        stop_commands = 3
        for i in range(stop_commands):
            self.device_send(KentaurDeviceCommand())
            rospy.loginfo(f"Sent stop command ({i+1}/{stop_commands})")

    @override
    def reset(self, state: KentaurDeviceState):
        self.control_law.clear()
        
        _, homeXy = self._device.state_to_home(state)
        
        homeX_r_init = homeXy @ state.state_yumi.pose_gripper_r
        homeX_r_final = homeX_r_init + self.diff_r
        self.trajectory_r.update(
            PoseParam(homeX_r_init.pos, homeX_r_init.rot, homeX_r_init.vel), 
            PoseParam(homeX_r_final.pos, homeX_r_final.rot, homeX_r_final.vel), 
            5)
        
        homeX_l_init = homeXy @ state.state_yumi.pose_gripper_l
        homeX_l_final = homeX_l_init + self.diff_l
        self.trajectory_l.update(
            PoseParam(homeX_l_init.pos, homeX_l_init.rot, homeX_l_init.vel), 
            PoseParam(homeX_l_final.pos, homeX_l_final.rot, homeX_l_final.vel), 
            5)
        
        self.trajectory_initial_time = rospy.Time.now()
    
    @override
    def fallback(self, state: KentaurDeviceState) -> KentaurDeviceAction:
        action = KentaurDeviceAction()
        action.action_yumi.control_space(MixedVelocityYumiAction.ControlSpace.JOINT_SPACE)
        action.action_yumi.velocity_joints(np.zeros(YumiRobotConstants.DOF))
        action.action_sleipner.twist_SE2 = np.zeros(3)
    
    @override
    def policy(self, state: KentaurDeviceState) -> KentaurDeviceAction:
        # timing
        real_now = rospy.Time.now()
        state_now: rospy.Time = state.state_sleipner.time
        dt = (real_now - state_now).to_sec()
        traj_time = (real_now - self.trajectory_initial_time).to_sec()
        
        homeXs, homeXy = self._device.state_to_home(state)
        
        # transform yumi grippers from yumi base to home
        curr_homeX_r = homeXy @ state.state_yumi.pose_gripper_r
        curr_homeX_l = homeXy @ state.state_yumi.pose_gripper_l
        
        
        yW_r = state.state_yumi.pose_wrench_r
        yW_l = state.state_yumi.pose_wrench_l
        
        if np.linalg.norm(yW_r) < 0.75:
            yW_r *= 0
        if np.linalg.norm(yW_l) < 0.75:
            yW_l *= 0
        
        curr_homeW_r = homeXy.inv().reactTo(yW_r)
        curr_homeW_l = homeXy.inv().reactTo(yW_l)
        
        current_param = YumiParam(
            curr_homeX_r.pos, curr_homeX_r.rot, curr_homeX_r.vel, 0, 
            curr_homeX_l.pos, curr_homeX_l.rot, curr_homeX_l.vel, 0)
        current_state = YumiParam_to_YumiCoordinatedRobotState(current_param)
        current_state._right.effector_wrc = curr_homeW_r
        current_state._left.effector_wrc = curr_homeW_l
        
        # gripper poses are already in home
        desired_pose_r = self.trajectory_r.compute(traj_time)
        desired_pose_l = self.trajectory_l.compute(traj_time)
        desired_param = YumiParam(
            desired_pose_r.pos, desired_pose_r.rot, desired_pose_r.vel, 0, 
            desired_pose_l.pos, desired_pose_l.rot, desired_pose_l.vel, 0)
        desired_state = YumiParam_to_YumiCoordinatedRobotState(desired_param)
        
        
        # current_state._right.effector_wrc = np.zeros(6)
        # current_state._left.effector_wrc = np.zeros(6)
        
        # compute target velocities
        vel_r, vel_l = self.control_law.update_and_compute(current_state, desired_state, dt)
        
        print("NOW RIGHT", curr_homeX_r)
        print("DES RIGHT", PoseParam_to_Frame(desired_pose_r))
        print()
        print("NOW LEFT ", curr_homeX_l)
        print("DES LEFT ", PoseParam_to_Frame(desired_pose_l))
        print()
        with np.printoptions(precision=3, suppress=True, floatmode='fixed'):
            print(vel_r, vel_l)
        print()
        print()
        
        # calculate new target velocities for this time step
        action = KentaurDeviceAction()
        action.action_yumi.control_space(MixedVelocityYumiAction.ControlSpace.INDIVIDUAL)
        action.action_yumi.timestep(dt)
        action.action_yumi.velocity_right(vel_r)
        action.action_yumi.velocity_left(vel_l)
        return action

    @override
    def solve_action(self, state: KentaurDeviceState, action: KentaurDeviceAction) -> KentaurDeviceCommand:
        
        # base to world transform (odometry/slam data)
        _, homeXy = self._device.state_to_home(state)
        
        # # base to world rotation derivative (in angle variable)
        # angle_z = np.linalg.norm(quat.as_rotation_vector(homeXb.rot))
        # home_dR_b = np.array([[-np.sin(angle_z), -np.cos(angle_z), 0],
        #                       [ np.cos(angle_z), -np.sin(angle_z), 0],
        #                       [               0,                0, 0]])
        
        # kinematics of yumi
        # yP_r = state.state_yumi.pose_gripper_r.pos
        # yR_r = quat.as_rotation_matrix(state.state_yumi.pose_gripper_r.rot)
        # yP_l = state.state_yumi.pose_gripper_l.pos
        # yR_l = quat.as_rotation_matrix(state.state_yumi.pose_gripper_l.rot)
        
        # bRy = np.kron(np.eye(2), self._device.bRy)  # blk_diag trick
        # w_dR_b = np.kron(np.eye(2), dR_b2w)
        
        # task jacobian
        homeJy_r = jacobian_change_base_frame(homeXy.rot, state.state_yumi.jacobian_gripper_r)
        homeJs_r = self._device.homeJ_s
        # homeJb_r = np.zeros((6, 3))
        # homeJb_r[:2,:2] = np.eye(2)
        # homeJb_r[:3, 2] = home_dR_b @ (self._device.bRy @ yP_r + self._device.bT_y)
        # homeJb_r[3:, 2] = home_dR_b @ (self._device.bRy @ yR_r)  # TODO must be a column vector, not a matrix
        
        homeJy_l = jacobian_change_base_frame(homeXy.rot, state.state_yumi.jacobian_gripper_l)
        homeJs_l = self._device.homeJ_s
        # homeJb_l = np.zeros((6, 3))
        # homeJb_l[:2,:2] = np.eye(2)
        # homeJb_l[:3, 2] = home_dR_b @ (self._device.bRy @ yP_l + self._device.bT_y)
        # homeJb_l[3:, 2] = home_dR_b @ (self._device.bRy @ yR_l)  # TODO must be a column vector, not a matrix
        
        
        balance = 0.0  # 1 is all yumi, 0 is all sleipner
        
        w = balance / (balance**2 + (1-balance)**2)
        w_ = (1-balance) / (balance**2 + (1-balance)**2)
        homeJ = np.zeros((6+6,7+7+3))
        # right arm
        homeJ[ 0:6, 0:7 ] = w * homeJy_r
        homeJ[ 0:6,14:17] = w_ * homeJs_r
        # left arm
        homeJ[6:12, 7:14] = w * homeJy_l
        homeJ[6:12,14:17] = w_ * homeJs_l
        
        
        # compute joints command
        vel_r = action.action_yumi["velocity_right"]
        vel_l = action.action_yumi["velocity_left"]
        vel_tgt = np.concatenate([vel_r, vel_l])  # cartesian, in home frame
        
        dq_target = np.linalg.pinv(homeJ) @ vel_tgt
        
        # log joints with clipping velocities
        dq_r_clip = np.abs(dq_target[0:7]) > YumiRobotConstants.JOINT_VEL_AB
        dq_l_clip = np.abs(dq_target[7:14]) > YumiRobotConstants.JOINT_VEL_AB
        if np.any(dq_r_clip) or np.any(dq_l_clip):
            idxs = np.arange(7) + 1
            labels = " ".join([f"R{i}" for i in idxs[dq_r_clip]] + [f"L{i}" for i in idxs[dq_l_clip]])
            print(f"Joints [ {labels} ] are clipping!")
        
        # create command
        command = KentaurDeviceCommand(
            YumiDualDeviceCommand(
                dq_target[0:14],  # [right, left]
                action.action_yumi.get("gripper_right"),
                action.action_yumi.get("gripper_left")), 
            SleipnerCartesianDeviceCommand(
                dq_target[14:17]))
        
        return command
    

if __name__ == "__main__":
    # starting ROS node
    rospy.init_node("kentaur_whole_body_controller", anonymous=False)
    
    device = KentaurDevice()
    controller = KentaurController(device)
    rospy.on_shutdown(controller.stop)
    
    controller.ready()
    controller.start()  # locking
