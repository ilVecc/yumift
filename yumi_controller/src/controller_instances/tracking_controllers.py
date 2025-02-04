#!/usr/bin/env python3
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).absolute().parent.parent))
import argparse

from typing import Tuple

import rospy
import numpy as np
import quaternion as quat

from yumi_controller.msg import YumiPosture as YumiPostureMsg

from core.controller_base import YumiDualController
from core.control_laws import YumiIndividualCartesianVelocityControlLaw
from core.trajectory import YumiParam
import core.msg_utils as msg_utils

from gains import GAINS


class YumiIndividualTrackingController(YumiDualController):
    """ Class for running tracking control using an instance of `YumiDualController`.
        Postures are sent with ROS Message `YumiPosture` and tracked using the 
        `YumiIndividualCartesianVelocityControlLaw` control law.
    """
    def __init__(self):
        super().__init__(iksolver="pinv")
        
        # define control law
        self.control_law = YumiIndividualCartesianVelocityControlLaw(GAINS)
        
        # prepare trajectory buffer
        self.desired_posture = YumiParam()
        
    def reset(self):
        """ Reinitialize the controller setting the current posture as desired posture. 
            This happens after EGM (re)connects
        """
        self.control_law.mode = "individual"
        self.effective_mode = self.control_law.mode
        # read current state of Yumi
        while True:
            self.fetch_device_status()
            if self.is_device_ready():
                self.desired_posture = YumiParam(
                    self.yumi_state.pose_gripper_r.pos, self.yumi_state.pose_gripper_r.rot, np.zeros(6), 0, 
                    self.yumi_state.pose_gripper_l.pos, self.yumi_state.pose_gripper_l.rot, np.zeros(6), 0)
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
        return quat.from_float_array(np.asarray(rot)) if rot else quat.one
    
    @staticmethod
    def _sanitize_vel(vel: Tuple[float]):
        return np.asarray(vel) if vel else np.array([0,0,0,0,0,0])
    
    def policy(self):
        """ Calculate target velocity for the current time step.
        """
        # update the current and desired robot state in the control law class, 
        # then compute the required command
        dt = (rospy.Time.now() - self.yumi_time).to_sec()
        yumi_desired_state = msg_utils.YumiParam_to_YumiCoordinatedRobotState(self.desired_posture)
        vel_r, vel_l = self.control_law.update_and_compute(self.yumi_state, yumi_desired_state, dt)
        
        # calculate new target velocities for this time step
        action = dict()
        action["control_space"] = self.control_law.mode
        action["right_velocity"], action["left_velocity"] = vel_r, vel_l
                
        return action


class SingleTrackingController(YumiIndividualTrackingController):
    def __init__(self):
        super().__init__()
        # listen for posture commands
        rospy.Subscriber("/posture_r", YumiPostureMsg, self._callback_posture, "right", queue_size=1, tcp_nodelay=False)
        rospy.Subscriber("/posture_l", YumiPostureMsg, self._callback_posture, "left", queue_size=1, tcp_nodelay=False)
    
    def _callback_posture(self, data: YumiPostureMsg, side: str):
        """ Gets called when a posture is received.  
        """
        if side == "right":
            self.desired_posture.pose_right.pos = self._sanitize_pos(data.positionRight)
            self.desired_posture.pose_right.rot = self._sanitize_rot(data.orientationRight)
            self.desired_posture.pose_right.vel = self._sanitize_vel(data.velocityRight)
            self.desired_posture.grip_right = data.gripperRight
        else:
            self.desired_posture.pose_left.pos = self._sanitize_pos(data.positionLeft)
            self.desired_posture.pose_left.rot = self._sanitize_rot(data.orientationLeft)
            self.desired_posture.pose_left.vel = self._sanitize_vel(data.velocityLeft)
            self.desired_posture.grip_left = data.gripperLeft

class WholeTrackingController(YumiIndividualTrackingController):
    def __init__(self):
        super().__init__()
        # listen for posture command for the overall robot
        rospy.Subscriber("/posture", YumiPostureMsg, self._callback_posture, queue_size=1, tcp_nodelay=False)

    def _callback_posture(self, data: YumiPostureMsg):
        """ Gets called when a posture is received.  
        """
        self.desired_posture.pose_right.pos = self._sanitize_pos(data.positionRight)
        self.desired_posture.pose_right.rot = self._sanitize_rot(data.orientationRight)
        self.desired_posture.pose_right.vel = self._sanitize_vel(data.velocityRight)
        self.desired_posture.grip_right = data.gripperRight
        self.desired_posture.pose_left.pos = self._sanitize_pos(data.positionLeft)
        self.desired_posture.pose_left.rot = self._sanitize_rot(data.orientationLeft)
        self.desired_posture.pose_left.vel = self._sanitize_vel(data.velocityLeft)
        self.desired_posture.grip_left = data.gripperLeft


def main():
    
    parser = argparse.ArgumentParser("Various tracking controllers")
    parser.add_argument("type", nargs="?", choices=["single", "whole"], default="single", type=str)
    args = parser.parse_args()
    
    # starting ROS node
    rospy.init_node("tracking_controllers", anonymous=False)
    
    if args.type == "single":
        yumi_controller = SingleTrackingController()
    elif args.type == "whole":
        yumi_controller = WholeTrackingController()
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
