#!/usr/bin/env python3
from typing_extensions import override

import rospy
import numpy as np, quaternion as quat

from dynamicals.utils import Frame, jacobian_change_base_frame
from dynamicals.impl import AbstractROSController
from pathfinder import PoseParam, CubicPoseTrajectory

from yumift_common.constants import YumiRobotConstants
from yumift_controllers.common.control_laws import YumiIndividualCartesianVelocityControlLaw
from yumift_controllers.impl.trajectory import YumiParam
from yumift_controllers.misc.utils import YumiCoordinatedRobotState_from_YumiParam

from kentaur_controllers.misc.utils import load_config
from kentaur_controllers.common.parameters import ControllerParameters
from kentaur_controllers.common import (
    KentaurDeviceState, KentaurDeviceCommand, KentaurDevice,
    YumiDualDeviceCommand, SleipnerCartesianDeviceCommand,
    KentaurDeviceAction, MixedVelocityYumiAction
)


class KentaurController(AbstractROSController[KentaurDeviceState, KentaurDeviceAction, KentaurDeviceCommand]):

    def __init__(self, kentaur_device : KentaurDevice):
        self._device : KentaurDevice
        super().__init__(kentaur_device)
        self.control_law = YumiIndividualCartesianVelocityControlLaw(load_config("gains_whole_body.yaml"))
        self.trajectory_r = CubicPoseTrajectory()
        self.trajectory_l = CubicPoseTrajectory()
        self.trajectory_initial_time = rospy.Time.now()
        
        # reset gripper position difference, in home frame
        # self.target_r = Frame(np.array([-0.2, -0.2, 0]))
        # self.target_l = Frame(np.array([-0.3, +0.1, 0]))
        self.target_r = Frame(np.array([-0.2, +0.2, 0]))
        self.target_l = Frame(np.array([-0.2, -0.2, 0]))
        self.target_time = 5.0
    
    @override
    def reset(self, state: KentaurDeviceState):
        self.control_law.clear()
        
        _, homeXy = self._device.state_to_home(state)
        
        homeX_r_init = homeXy @ state.state_yumi.pose_gripper_r
        homeX_r_final = self.target_r @ homeX_r_init
        self.trajectory_r.update(PoseParam.from_Frame(homeX_r_init), PoseParam.from_Frame(homeX_r_final), self.target_time)
        
        homeX_l_init = homeXy @ state.state_yumi.pose_gripper_l
        homeX_l_final = self.target_l @ homeX_l_init
        self.trajectory_l.update(PoseParam.from_Frame(homeX_l_init), PoseParam.from_Frame(homeX_l_final), self.target_time)
        
        self.trajectory_initial_time = rospy.Time.now()
    
    @override
    def fallback(self, state: KentaurDeviceState) -> KentaurDeviceAction:
        action = KentaurDeviceAction()
        action.action_yumi.control_space(MixedVelocityYumiAction.ControlSpace.JOINT_SPACE)
        action.action_yumi.velocity_joints(np.zeros(YumiRobotConstants.DOF))
    
    @override
    def policy(self, state: KentaurDeviceState) -> KentaurDeviceAction:
        ctrl_dt = ControllerParameters.dt  # self.dt(state.state_sleipner.time)
        traj_dt = self.dt(self.trajectory_initial_time)
        
        homeXs, homeXy = self._device.state_to_home(state)
        
        # transform yumi grippers from yumi base to home
        curr_homeX_r = homeXy @ state.state_yumi.pose_gripper_r
        curr_homeX_l = homeXy @ state.state_yumi.pose_gripper_l
        curr_param = YumiParam.from_PoseParams(curr_homeX_r, curr_homeX_l)
        curr_state = YumiCoordinatedRobotState_from_YumiParam(curr_param)
        
        # gripper poses are already in home
        des_homeX_r = self.trajectory_r.compute(traj_dt)
        des_homeX_l = self.trajectory_l.compute(traj_dt)
        des_param = YumiParam.from_PoseParams(des_homeX_r, des_homeX_l)
        des_state = YumiCoordinatedRobotState_from_YumiParam(des_param)
        
        # compute target velocities
        vel_r, vel_l = self.control_law.update_and_compute(curr_state, des_state, ctrl_dt)
        
        # calculate new target velocities for this time step
        action = KentaurDeviceAction()
        action.action_yumi.control_space(MixedVelocityYumiAction.ControlSpace.INDIVIDUAL)
        action.action_yumi.timestep(ctrl_dt)
        action.action_yumi.velocity_right(vel_r)
        action.action_yumi.velocity_left(vel_l)
        return action

    @override
    def solve_action(self, state: KentaurDeviceState, action: KentaurDeviceAction) -> KentaurDeviceCommand:
        
        # base to world transform (odometry/slam data)
        homeJ = self._device.jacobian_to_home(state, alpha=2.75)
        
        # compute joints command
        vel_tgt = np.zeros(12)  # cartesian, in home frame
        vel_tgt[0:6] = action.action_yumi["velocity_right"]
        vel_tgt[6:12] = action.action_yumi["velocity_left"]
        
        dq_target = np.linalg.pinv(homeJ) @ vel_tgt
        
        # log joints with clipping velocities
        dq_r_clip = np.abs(dq_target[0:7]) > YumiRobotConstants.JOINT_VEL_AB
        dq_l_clip = np.abs(dq_target[7:14]) > YumiRobotConstants.JOINT_VEL_AB
        if np.any(dq_r_clip) or np.any(dq_l_clip):
            idxs = np.arange(7) + 1
            labels = " ".join([f"R{i}" for i in idxs[dq_r_clip]] + [f"L{i}" for i in idxs[dq_l_clip]])
            rospy.logwarn(f"Joints [ {labels} ] are clipping!")
        
        # create command
        command = KentaurDeviceCommand(
            YumiDualDeviceCommand(dq_target[0:14],  # [right, left]
                                  action.action_yumi.get("gripper_right"),
                                  action.action_yumi.get("gripper_left")), 
            SleipnerCartesianDeviceCommand(dq_target[14:17]))
        
        return command
    

if __name__ == "__main__":
    # starting ROS node
    rospy.init_node("kentaur_whole_body_controller", anonymous=False)
    
    device = KentaurDevice()
    controller = KentaurController(device)
    
    controller.ready()
    controller.start(ControllerParameters.update_freq)  # locking
