#!/usr/bin/env python3
import argparse
from typing_extensions import override

import rospy
import numpy as np, quaternion as quat

from yumift_msgs.msg import YumiPosture as YumiPostureMsg

from yumift_controllers.common.device import YumiDualDeviceState, YumiDevice
from yumift_controllers.common.controller_base import YumiDualController, MixedVelocityYumiAction
from yumift_controllers.common.control_laws import YumiIndividualCartesianVelocityControlLaw, YumiDualCartesianVelocityControlLaw
from yumift_controllers.impl.trajectory import YumiParam
from yumift_controllers.misc.utils import (
    load_config, sanitize_pos, sanitize_rot, sanitize_vel, sanitize_grip,
    YumiCoordinatedRobotState_from_YumiParam
)


class YumiIndividualTrackingController(YumiDualController):
    """ Class for running tracking control using an instance of `YumiDualController`.
        Postures are sent with ROS Message `YumiPosture` and tracked using the 
        `YumiIndividualCartesianVelocityControlLaw` control law.
    """
    def __init__(self, gains):
        super().__init__(yumi_device=YumiDevice(), ikalgorithms="pinv")
        # define control law
        self.control_law = YumiIndividualCartesianVelocityControlLaw(gains)
        # prepare trajectory buffer
        self.desired_posture = YumiParam()
    
    @override
    def reset(self, state: YumiDualDeviceState):
        """ Reinitialize the controller setting the current posture as desired posture. 
            This happens after EGM (re)connects
        """
        # HACK must be changed to correct Enum
        self.control_law.mode = YumiDualCartesianVelocityControlLaw.ControlMode.INDIVIDUAL
        self.effective_mode = self.control_law.mode
        # read current state of Yumi
        while not self.device_is_ready():
            rospy.logwarn("Controller cannot be reset (Yumi is not ready, retrying in 5 seconds)")
            rospy.sleep(5)
        self.desired_posture = YumiParam.from_Frames(state.pose_gripper_r.motionless(), state.pose_gripper_l.motionless(), 0, 0)
        rospy.loginfo("Controller reset (previous posture has been discarded)")
    
    @override
    def policy(self, state: YumiDualDeviceState) -> MixedVelocityYumiAction:
        """ Calculate target velocity for the current time step.
        """
        # update the current and desired robot state in the control law class, 
        # then compute the required command
        dt = self.dt(state.time)
        yumi_desired_state = YumiCoordinatedRobotState_from_YumiParam(self.desired_posture)
        vel_r, vel_l = self.control_law.update_and_compute(state, yumi_desired_state, dt)
        
        # calculate new target velocities for this time step
        action = MixedVelocityYumiAction()
        action.control_space(MixedVelocityYumiAction.ControlSpace.from_str(self.control_law.mode))
        action.timestep(dt)
        action.velocity_right(vel_r) 
        action.velocity_left(vel_l)
        return action


class SingleTrackingController(YumiIndividualTrackingController):
    def __init__(self, gains):
        super().__init__(gains)
        # listen for posture commands
        rospy.Subscriber("/posture_r", YumiPostureMsg, self._callback_posture, "right", queue_size=1)
        rospy.Subscriber("/posture_l", YumiPostureMsg, self._callback_posture, "left", queue_size=1)
    
    def _callback_posture(self, posture: YumiPostureMsg, side: str):
        """ Gets called when a posture is received.  
        """
        if side == "right":
            self.desired_posture.pose_right.pos = sanitize_pos(posture.pose_primary.position)
            self.desired_posture.pose_right.rot = sanitize_rot(posture.pose_primary.orientation)
            self.desired_posture.pose_right.vel = sanitize_vel(posture.twist_primary)
            self.desired_posture.grip_right = sanitize_grip(posture.gripper_right)
        elif side == "left":
            self.desired_posture.pose_left.pos = sanitize_pos(posture.pose_secondary.position)
            self.desired_posture.pose_left.rot = sanitize_rot(posture.pose_secondary.orientation)
            self.desired_posture.pose_left.vel = sanitize_vel(posture.twist_secondary)
            self.desired_posture.grip_left = sanitize_grip(posture.gripper_left)
        else:
            raise Exception(f"No such side {side}")

class WholeTrackingController(YumiIndividualTrackingController):
    def __init__(self, gains):
        super().__init__(gains)
        # listen for posture command for the overall robot
        rospy.Subscriber("/posture", YumiPostureMsg, self._callback_posture, queue_size=1)

    def _callback_posture(self, posture: YumiPostureMsg):
        """ Gets called when a posture is received.  
        """
        self.desired_posture.pose_right.pos = sanitize_pos(posture.pose_primary.position)
        self.desired_posture.pose_right.rot = sanitize_rot(posture.pose_primary.orientation)
        self.desired_posture.pose_right.vel = sanitize_vel(posture.twist_primary)
        self.desired_posture.grip_right = sanitize_grip(posture.gripper_right)
        self.desired_posture.pose_left.pos = sanitize_pos(posture.pose_secondary.position)
        self.desired_posture.pose_left.rot = sanitize_rot(posture.pose_secondary.orientation)
        self.desired_posture.pose_left.vel = sanitize_vel(posture.twist_secondary)
        self.desired_posture.grip_left = sanitize_grip(posture.gripper_left)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("Various tracking controllers")
    parser.add_argument("type", nargs="?", choices=["single", "whole"], default="single", type=str)
    args = parser.parse_args()
    
    # starting ROS node
    rospy.init_node("tracking_controllers", anonymous=False)
    
    if args.type == "single":
        yumi_controller = SingleTrackingController(load_config("gains_simple.yaml"))
    elif args.type == "whole":
        yumi_controller = WholeTrackingController(load_config("gains_simple.yaml"))
    else:
        raise AttributeError(f"no such option '{parser.type}'")
    
    yumi_controller.ready()
    yumi_controller.start()  # locking
