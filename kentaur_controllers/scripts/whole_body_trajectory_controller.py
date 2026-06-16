#!/usr/bin/env python3
from typing_extensions import override

import threading

import rospy
import numpy as np

from yumift_msgs.helper import Helper
from yumift_msgs.msg import YumiTrajectory as YumiTrajectoryMsg, YumiPosture as YumiPostureMsg

from dynamicals.utils import Frame
from dynamicals.impl import AbstractROSController

from yumift_common.constants import YumiRobotConstants
from yumift_controllers.common.control_laws import YumiIndividualCartesianVelocityControlLaw
from yumift_controllers.ik.pinv_tasks import secondary_neutral
from yumift_controllers.impl.trajectory import PoseParam, YumiParam, YumiTrajectory, YumiTrajectoryParam
from yumift_controllers.misc.utils import YumiCoordinatedRobotState_from_YumiParam

from kentaur_controllers.misc.utils import load_config
from kentaur_controllers.common.parameters import ControllerParameters
from kentaur_controllers.common import (
    KentaurDeviceState, KentaurDeviceCommand, KentaurDevice,
    YumiDualDeviceCommand, SleipnerCartesianDeviceCommand,
    KentaurDeviceAction, MixedVelocityYumiAction
)


class KentaurTrajectoryController(AbstractROSController[KentaurDeviceState, KentaurDeviceAction, KentaurDeviceCommand]):

    def __init__(self, kentaur_device : KentaurDevice):
        self._device : KentaurDevice
        super().__init__(kentaur_device)
        self.control_law = YumiIndividualCartesianVelocityControlLaw(load_config("gains_whole_body_trajectory.yaml"))
        self.trajectory = YumiTrajectory()
        self.trajectory_initial_time = rospy.Time.now()
        self._lock_trajectory = threading.Lock()
        
        # listen for trajectory commands
        trajectory_topic = "/trajectory"
        rospy.Subscriber(trajectory_topic, YumiTrajectoryMsg, self._callback_trajectory, queue_size=1)
    
    @override
    def reset(self, state: KentaurDeviceState):
        # wait for Yumi
        while not self.device_is_ready():
            rospy.logwarn("Controller cannot be reset (Yumi is not ready, retrying in 5 seconds)")
            rospy.sleep(5)
        # create the dummy path
        _, homeXy = self._device.state_to_home(state)
        homeX_r_init = homeXy @ state.state_yumi.pose_gripper_r
        homeX_l_init = homeXy @ state.state_yumi.pose_gripper_l
        current_pose = YumiParam.from_Frames(homeX_r_init.motionless(), homeX_l_init.motionless(), 0, 0)
        path = [YumiTrajectoryParam(current_pose, 0), YumiTrajectoryParam(current_pose, 0.00001)]
        # update the trajectory
        with self._lock_trajectory:
            self.control_law.clear()
            self.trajectory.update(path)
            self.trajectory_initial_time = rospy.Time.now()
        rospy.loginfo("Controller reset (previous trajectory has been discarded)")
        rospy.loginfo(f"ODOM   in [ -- ]: {state.state_sleipner.to_Frame()}")
        rospy.loginfo(f"YUMI   in [HOME]: {homeXy}")
        rospy.loginfo(f"tool_r in [HOME]: {homeX_r_init}")
        rospy.loginfo(f"tool_l in [HOME]: {homeX_l_init}")
    
    def _callback_trajectory(self, traj_msg: YumiTrajectoryMsg):
        """ Path point are expected in HOME frame
        """
        ########################   PREPARE TRAJECTORY   #######################
        # use current position, rotation and velocity as first trajectory points
        # use the required mode as first pose
        curr_state = self.device_read()
        _, homeXy = self._device.state_to_home(curr_state)
        curr_homeX_r = homeXy @ curr_state.state_yumi.pose_gripper_r
        curr_homeX_l = homeXy @ curr_state.state_yumi.pose_gripper_l
        trajectory = Helper.decode_trajectory((curr_homeX_r, curr_homeX_l, curr_state.state_yumi.grip_r, curr_state.state_yumi.grip_l), traj_msg)
        trajectory = [YumiTrajectoryParam(YumiParam.from_Frames(frame_1, frame_2, grip_r, grip_l), duration)
                      for (frame_1, frame_2, grip_r, grip_l), duration in trajectory]
        #######################################################################
        
        # update the trajectory
        with self._lock_trajectory:
            self.trajectory.update(trajectory)
            self.trajectory_initial_time = rospy.Time.now()
        rospy.loginfo(f"New trajectory received")
    
    @override
    def policy(self, state: KentaurDeviceState) -> KentaurDeviceAction:
        ctrl_dt = ControllerParameters.dt  # self.dt(state.state_sleipner.time)
        traj_dt = self.dt(self.trajectory_initial_time)
        
        _, homeXy = self._device.state_to_home(state)
        
        # transform yumi grippers from yumi base to home
        curr_homeX_r = homeXy @ state.state_yumi.pose_gripper_r
        curr_homeX_l = homeXy @ state.state_yumi.pose_gripper_l
        curr_param = YumiParam.from_Frames(curr_homeX_r, curr_homeX_l)
        curr_state = YumiCoordinatedRobotState_from_YumiParam(curr_param)
        
        # gripper poses are already in home
        with self._lock_trajectory:
            des_homeX = self.trajectory.compute(traj_dt)
        des_param = YumiParam.from_PoseParams(des_homeX.pose_right, des_homeX.pose_left)
        des_state = YumiCoordinatedRobotState_from_YumiParam(des_param)
        
        try:
            vel_r, vel_l = self.control_law.update_and_compute(curr_state, des_state, ctrl_dt)
            
            action = KentaurDeviceAction()
            action.action_yumi.control_space(MixedVelocityYumiAction.ControlSpace.INDIVIDUAL)
            action.action_yumi.timestep(ctrl_dt)
            action.action_yumi.velocity_right(vel_r)
            action.action_yumi.velocity_left(vel_l)
            
        except Exception as ex:
            rospy.logerr(f"Stopping motion (exception: {ex})")
            action = self.fallback(state)
        
        return action

    @override
    def solve_action(self, state: KentaurDeviceState, action: KentaurDeviceAction) -> KentaurDeviceCommand:
        
        # base to world transform (odometry/slam data)
        homeJ = self._device.jacobian_to_home(state, alpha=2.75)
        
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
    rospy.init_node("kentaur_whole_body_trajectory_controller", anonymous=False)
    
    device = KentaurDevice()
    controller = KentaurTrajectoryController(device)
    
    controller.ready()
    controller.start(ControllerParameters.dt)  # locking
