#!/usr/bin/env python3
import argparse

from typing import Tuple

import rospy
import numpy as np, quaternion as quat

from yumift_msgs.msg import YumiPosture as YumiPostureMsg

from yumift_controllers.common.device import YumiDualDeviceState, YumiDevice
from yumift_controllers.common.controller_base import YumiDualController, MixedVelocityYumiAction
from yumift_controllers.common.control_laws import YumiIndividualCartesianVelocityControlLaw
from yumift_controllers.impl.trajectory import YumiParam
from yumift_controllers.misc.utils import load_config, YumiParam_to_YumiCoordinatedRobotState


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
        
    def reset(self, state: YumiDualDeviceState):
        """ Reinitialize the controller setting the current posture as desired posture. 
            This happens after EGM (re)connects
        """
        # BUG must be changed to correct Enum
        self.control_law.mode = "individual"
        self.effective_mode = self.control_law.mode
        # read current state of Yumi
        while True:
            self.device_read()
            if self.device_is_ready():
                self.desired_posture = YumiParam(
                    state.pose_gripper_r.pos, state.pose_gripper_r.rot, np.zeros(6), 0, 
                    state.pose_gripper_l.pos, state.pose_gripper_l.rot, np.zeros(6), 0)
                print("Controller reset (previous posture has been discarded)")
                break
            else:
                print("Controller cannot be reset (Yumi is not ready, retrying in 5 seconds)")
                rospy.sleep(5)
    
    @staticmethod
    def _sanitize_pos(pos: Tuple[float]):
        return np.asarray(pos) if pos else np.array([0,0,0])
    
    @staticmethod
    def _sanitize_rot(rot: Tuple[float]):
        return quat.quaternion(*rot) if rot else quat.one
    
    @staticmethod
    def _sanitize_vel(vel: Tuple[float]):
        return np.asarray(vel) if vel else np.array([0,0,0,0,0,0])
    
    def policy(self, state: YumiDualDeviceState) -> MixedVelocityYumiAction:
        """ Calculate target velocity for the current time step.
        """
        # update the current and desired robot state in the control law class, 
        # then compute the required command
        dt = (rospy.Time.now() - state.time).to_sec()
        yumi_desired_state = YumiParam_to_YumiCoordinatedRobotState(self.desired_posture)
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
        rospy.Subscriber("/posture_r", YumiPostureMsg, self._callback_posture, "right", queue_size=1, tcp_nodelay=False)
        rospy.Subscriber("/posture_l", YumiPostureMsg, self._callback_posture, "left", queue_size=1, tcp_nodelay=False)
    
    def _callback_posture(self, posture: YumiPostureMsg):
        """ Gets called when a posture is received.  
        """
        self.desired_posture.pose_right.pos = self._sanitize_pos(posture.pose_primary.position)
        self.desired_posture.pose_right.rot = self._sanitize_rot(posture.pose_primary.orientation)
        self.desired_posture.pose_right.vel = self._sanitize_vel(posture.twist_primary)
        self.desired_posture.grip_right = posture.gripper_right
        self.desired_posture.pose_left.pos = self._sanitize_pos(posture.pose_secondary.position)
        self.desired_posture.pose_left.rot = self._sanitize_rot(posture.pose_secondary.orientation)
        self.desired_posture.pose_left.vel = self._sanitize_vel(posture.twist_secondary)
        self.desired_posture.grip_left = posture.gripper_right

class WholeTrackingController(YumiIndividualTrackingController):
    def __init__(self, gains):
        super().__init__(gains)
        # listen for posture command for the overall robot
        rospy.Subscriber("/posture", YumiPostureMsg, self._callback_posture, queue_size=1, tcp_nodelay=False)

    def _callback_posture(self, posture: YumiPostureMsg):
        """ Gets called when a posture is received.  
        """
        self.desired_posture.pose_right.pos = self._sanitize_pos(posture.pose_primary.position)
        self.desired_posture.pose_right.rot = self._sanitize_rot(posture.pose_primary.orientation)
        self.desired_posture.pose_right.vel = self._sanitize_vel(posture.twist_primary)
        self.desired_posture.grip_right = posture.gripper_right
        self.desired_posture.pose_left.pos = self._sanitize_pos(posture.pose_secondary.position)
        self.desired_posture.pose_left.rot = self._sanitize_rot(posture.pose_secondary.orientation)
        self.desired_posture.pose_left.vel = self._sanitize_vel(posture.twist_secondary)
        self.desired_posture.grip_left = posture.gripper_right


def main():
    
    parser = argparse.ArgumentParser("Various tracking controllers")
    parser.add_argument("type", nargs="?", choices=["single", "whole"], default="single", type=str)
    args = parser.parse_args()
    
    # starting ROS node
    rospy.init_node("tracking_controllers", anonymous=False)
    
    gains = load_config("gains.yaml")
    
    if args.type == "single":
        yumi_controller = SingleTrackingController(gains["CLIK"])
    elif args.type == "whole":
        yumi_controller = WholeTrackingController(gains["CLIK"])
    else:
        raise AttributeError(f"no such option '{parser.type}'")
    
    def shutdown_callback():
        print("Controller shutting down")
        yumi_controller.stop()
    
    rospy.on_shutdown(shutdown_callback)
    
    yumi_controller.ready()
    yumi_controller.start()  # locking


if __name__ == "__main__":
    main()
