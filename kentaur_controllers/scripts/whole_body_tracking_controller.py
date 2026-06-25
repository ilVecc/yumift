#!/usr/bin/env python3
from typing_extensions import override

import rospy
import numpy as np

from yumift_msgs.helper import Helper
from yumift_msgs.msg import YumiPosture as YumiPostureMsg

from dynamicals.impl import AbstractROSController

from yumift_controllers.common.control_laws import YumiIndividualCartesianVelocityControlLaw
from yumift_controllers.ik.pinv_tasks import secondary_neutral
from yumift_controllers.impl.trajectory import YumiParam
from yumift_controllers.misc.utils import YumiCoordinatedRobotState_from_YumiParam

from kentaur_controllers.misc.utils import load_config
from kentaur_controllers.common.parameters import ControllerParameters
from kentaur_controllers.common import (
    KentaurDeviceState, KentaurDeviceCommand, KentaurDevice,
    YumiDualDeviceCommand, SleipnerCartesianDeviceCommand,
    KentaurDeviceAction, MixedVelocityYumiAction
)


class KentaurTrackingController(AbstractROSController[KentaurDeviceState, KentaurDeviceAction, KentaurDeviceCommand]):

    def __init__(self, kentaur_device : KentaurDevice):
        self._device : KentaurDevice
        super().__init__(kentaur_device)
        self.control_law = YumiIndividualCartesianVelocityControlLaw(load_config("gains_whole_body_tracking.yaml"))
        self.target = YumiParam()
        
        # listen for trajectory commands
        rospy.Subscriber("/posture", YumiPostureMsg, self._callback_posture, queue_size=1)
    
    @override
    def reset(self, state: KentaurDeviceState):
        self.control_law.clear()
        # wait for Yumi
        while not self.device_is_ready():
            rospy.logwarn("Controller cannot be reset (Yumi is not ready, retrying in 5 seconds)")
            rospy.sleep(5)
        # create the dummy path
        homeX_r_init, homeX_l_init = self._device.grippers_wrt_home(state)
        self.target = YumiParam.from_Frames(homeX_r_init.motionless(), homeX_l_init.motionless(), 
                                            state.state_yumi.grip_r, state.state_yumi.grip_l)
        # print some usefult frames
        homeXs, homeXy = self._device.robots_wrt_home(state)
        rospy.loginfo("Controller reset (previous target has been discarded)")
        rospy.loginfo(f"ODOM   in [ -- ]: {state.state_sleipner.to_Frame()}")
        rospy.loginfo(f"ODOM   in [HOME]: {homeXs}")
        rospy.loginfo(f"YUMI   in [HOME]: {homeXy}")
        rospy.loginfo(f"tool_r in [HOME]: {homeX_r_init}")
        rospy.loginfo(f"tool_l in [HOME]: {homeX_l_init}")
    
    def _callback_posture(self, posture_msg: YumiPostureMsg):
        """ Posture is expected in HOME frame
        """
        self.target.pose_right.pos = Helper.sanitize_pos(posture_msg.pose_primary.position)
        self.target.pose_right.rot = Helper.sanitize_rot(posture_msg.pose_primary.orientation)
        self.target.pose_right.vel = Helper.sanitize_vel(posture_msg.twist_primary)
        self.target.grip_right = Helper.sanitize_grip(posture_msg.gripper_right)
        self.target.pose_left.pos = Helper.sanitize_pos(posture_msg.pose_secondary.position)
        self.target.pose_left.rot = Helper.sanitize_rot(posture_msg.pose_secondary.orientation)
        self.target.pose_left.vel = Helper.sanitize_vel(posture_msg.twist_secondary)
        self.target.grip_left = Helper.sanitize_grip(posture_msg.gripper_left)
    
    @override
    def policy(self, state: KentaurDeviceState) -> KentaurDeviceAction:
        ctrl_dt = ControllerParameters.dt  # self.dt(state.state_sleipner.time)
        
        # transform yumi grippers from yumi base to home
        curr_homeX_r, curr_homeX_l = self._device.grippers_wrt_home(state)
        curr_param = YumiParam.from_Frames(curr_homeX_r, curr_homeX_l)
        curr_state = YumiCoordinatedRobotState_from_YumiParam(curr_param)
        
        des_state = YumiCoordinatedRobotState_from_YumiParam(self.target)
        
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
        homeJ = self._device.jacobian_to_home(state, alpha=1.5)
        
        # compute joints command
        vel_tgt = np.zeros(12)  # cartesian, in home frame
        vel_tgt[0:6] = action.action_yumi["velocity_right"]
        vel_tgt[6:12] = action.action_yumi["velocity_left"]
        
        homeJ_pinv = np.linalg.pinv(homeJ)
        dq_target = homeJ_pinv @ vel_tgt
        dq_target[:14] += (np.eye(14) - (homeJ_pinv @ homeJ)[:14,:14]) @ secondary_neutral(state.state_yumi.joint_pos, None, k=30)
        
        # create command
        command = KentaurDeviceCommand(
            YumiDualDeviceCommand(dq_target[0:14],  # [right, left]
                                  action.action_yumi.get("gripper_right"),
                                  action.action_yumi.get("gripper_left")), 
            SleipnerCartesianDeviceCommand(dq_target[14:17]))
        
        return command
    

if __name__ == "__main__":
    rospy.init_node("kentaur_whole_body_tracking_controller", anonymous=False)
    
    device = KentaurDevice()
    controller = KentaurTrackingController(device)
    
    controller.ready()
    controller.start(ControllerParameters.update_freq)  # locking
