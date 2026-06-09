#!/usr/bin/env python3
from typing_extensions import override

import threading

import rospy
import numpy as np, quaternion as quat

from yumift_msgs.msg import YumiTrajectory as YumiTrajectoryMsg, YumiPosture as YumiPostureMsg

from dynamicals.utils import Frame, jacobian_change_base_frame
from dynamicals.impl import AbstractROSController

from yumift_common.constants import YumiRobotConstants
from yumift_controllers.common.control_laws import YumiIndividualCartesianVelocityControlLaw
from yumift_controllers.ik.pinv_tasks import secondary_neutral
from yumift_controllers.impl.trajectory import YumiParam, YumiTrajectory, YumiTrajectoryParam
from yumift_controllers.misc.utils import (
    sanitize_pos, sanitize_rot, sanitize_vel, sanitize_grip,
    YumiCoordinatedRobotState_from_YumiParam
)

from kentaur_controllers.misc.utils import load_config
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
    
    def _callback_trajectory(self, traj_msg: YumiTrajectoryMsg):
        """ Path point are expected in HOME frame
        """
        ########################   PREPARE TRAJECTORY   #######################
        # use current position, rotation and velocity as first trajectory points
        # use the required mode as first pose
        curr_state = self.device_read()
        _, homeXy = self._device.state_to_home(curr_state)
        curr_homeX_r, curr_homeX_l = homeXy @ curr_state.state_yumi.pose_gripper_r, homeXy @ curr_state.state_yumi.pose_gripper_r
        traj_point = YumiParam.from_Frames(curr_homeX_r, curr_homeX_l, curr_state.state_yumi.grip_r, curr_state.state_yumi.grip_l)
        trajectory = [YumiTrajectoryParam(traj_point, duration=0)]
        
        # append trajectory points from msg
        for posture in traj_msg.trajectory:
            posture: YumiPostureMsg
            pos_r = sanitize_pos(posture.pose_primary.position, default_none=False)
            rot_r = sanitize_rot(posture.pose_primary.orientation, default_none=False)
            vel_r = sanitize_vel(posture.twist_primary)
            grip_r = sanitize_grip(posture.gripper_right, default_none=False)
            frame_r = Frame(pos_r, rot_r) #, vel_1)
            pos_l = sanitize_pos(posture.pose_secondary.position, default_none=False)
            rot_l = sanitize_rot(posture.pose_secondary.orientation, default_none=False)
            vel_l = sanitize_vel(posture.twist_secondary)
            grip_l = sanitize_grip(posture.gripper_left, default_none=False)
            frame_l = Frame(pos_l, rot_l) #, vel_2)
            duration = posture.time_to_execute.to_sec()
            
            frame_r.vel, frame_r.vel = vel_r, vel_l
            traj_point = YumiParam.from_Frames(frame_r, grip_r, frame_l, grip_l)
            trajectory.append(YumiTrajectoryParam(traj_point, duration))
        #######################################################################
        
        # update the trajectory
        with self._lock_trajectory:
            self.trajectory.update(trajectory)
            self.trajectory_initial_time = rospy.Time.now()
        rospy.loginfo(f"New trajectory received")
    
    @override
    def policy(self, state: KentaurDeviceState) -> KentaurDeviceAction:
        ctrl_dt = self.dt(state.state_sleipner.time)
        traj_dt = self.dt(self.trajectory_initial_time)
        
        homeXs, homeXy = self._device.state_to_home(state)
        
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
        homeXs, homeXy = self._device.state_to_home(state)
        
        # task jacobian
        homeJy_r = jacobian_change_base_frame(homeXy.rot, state.state_yumi.jacobian_gripper_r)
        homeJs_r = jacobian_change_base_frame(homeXs.rot, state.state_sleipner.jacobian_SE3)
        homeJy_l = jacobian_change_base_frame(homeXy.rot, state.state_yumi.jacobian_gripper_l)
        homeJs_l = homeJs_r
        
        
        alpha = 2.75  # 1> is more yumi, 1< is more sleipner
        beta = 1/alpha
        
        homeJ = np.zeros((6+6,7+7+3))
        # right arm
        homeJ[ 0:6, 0:7 ] = alpha * homeJy_r
        homeJ[ 0:6,14:17] = beta * homeJs_r
        # left arm
        homeJ[6:12, 7:14] = alpha * homeJy_l
        homeJ[6:12,14:17] = beta * homeJs_l
        
        # compute joints command
        vel_r = action.action_yumi["velocity_right"]
        vel_l = action.action_yumi["velocity_left"]
        vel_tgt = np.concatenate([vel_r, vel_l])  # cartesian, in home frame
        
        
        homeJ_pinv = np.linalg.pinv(homeJ)
        dq_target = homeJ_pinv @ vel_tgt
        dq_target[:14] += (np.eye(14) - (homeJ_pinv @ homeJ)[:14,:14]) @ secondary_neutral(state.state_yumi.joint_pos, None, k=30)
        
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
    rospy.init_node("kentaur_whole_body_trajectory_controller", anonymous=False)
    
    device = KentaurDevice()
    controller = KentaurTrajectoryController(device)
    
    controller.ready()
    controller.start(250)  # locking
