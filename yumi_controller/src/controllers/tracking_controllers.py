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
from core.control_laws import YumiIndividualCartesianVelocityControlLaw, YumiDualCartesianVelocityControlLaw
from core.trajectory import YumiParam
import core.msg_utils as msg_utils

from gains import GAINS


class YumiIndividualTrackingController(YumiDualController):
    """ Class for running trajectory control using an instance of `YumiDualController`.
        Trajectory parameters are sent with ROS Message `YumiTrajectory` and from 
        those a `YumiTrajectory` is constructed and tracked using the chosen 
        `YumiDualCartesianVelocityControlLaw`.
    """
    def __init__(self, posture_r_topic: str, posture_l_topic: str):
        super().__init__(iksolver=self.IKSolver.PINV, symmetry=0.)
        
        # define control law
        self.control_law = YumiIndividualCartesianVelocityControlLaw(GAINS)
        
        # prepare trajectory buffer
        self.initial_time = rospy.Time.now()
        self.desired_posture = YumiParam()
        self.reset()  # init trajectory (set current position)
        
        # listen for trajectory commands
        rospy.Subscriber(posture_r_topic, YumiPostureMsg, self._callback_posture, "right", queue_size=1, tcp_nodelay=False)
        rospy.Subscriber(posture_l_topic, YumiPostureMsg, self._callback_posture, "left", queue_size=1, tcp_nodelay=False)
        
    def reset(self):
        """ Initialize the controller setting the current posture as desired posture. 
        """
        self.control_law.mode = "individual"
        self.effective_mode = self.control_law.mode
        self.desired_posture = YumiParam(
            self.yumi_state.pose_gripper_r.pos, self.yumi_state.pose_gripper_r.rot, np.zeros(6), 0, 
            self.yumi_state.pose_gripper_l.pos, self.yumi_state.pose_gripper_l.rot, np.zeros(6), 0)
        print("Controller reset (previous posture has been discarded)")
    
    @staticmethod
    def _sanitize_pos(pos: Tuple[float]):
        return np.asarray(pos) if pos else np.array([0,0,0])
    
    @staticmethod
    def _sanitize_rot(rot: Tuple[float]):
        return quat.from_float_array(np.asarray(rot)) if rot else quat.one
    
    @staticmethod
    def _sanitize_vel(vel: Tuple[float]):
        return np.asarray(vel) if vel else np.array([0,0,0,0,0,0])
    
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
        
        self.initial_time = rospy.Time.now()
    
    def policy(self):
        """ Calculate target velocity for the current time step.
        """
        
        # update timing information
        now = rospy.Time.now()
        dt = (now - self.timestamp).to_sec()
        self.control_law.update_current_dt(dt)
        
        # update pose and wrench for the control law class
        self.control_law.update_current_pose(self.yumi_state)
        
        # calculate new target velocities and positions for this time step
        yumi_target_state = msg_utils.YumiParam_to_YumiCoordinatedRobotState(self.desired_posture)
        
        self.control_law.update_target_pose(yumi_target_state)
        
        # CALCULATE VELOCITIES
        action = dict()
        # set velocities
        action["control_space"] = self.control_law.mode
        action["right_velocity"], action["left_velocity"] = self.control_law.compute_target_velocity()
                
        return action


class SimpleTrackingController(YumiIndividualTrackingController):
    def __init__(self):
        super().__init__("/posture")


def main():
    
    parser = argparse.ArgumentParser("Various tracking controllers")
    parser.add_argument("type", nargs="?", choices=["simple"], default="simple", type=str)
    args = parser.parse_args()
    
    # starting ROS node
    rospy.init_node("tracking_controllers", anonymous=False)
    
    if args.type == "simple":
        yumi_controller = SimpleTrackingController()
    else:
        raise AttributeError(f"no such option '{parser.type}'")
    
    def shutdown_callback():
        yumi_controller.pause()
        print("Controller shutting down")
    
    rospy.on_shutdown(shutdown_callback)
    
    
    yumi_controller.start()  # locking


if __name__ == "__main__":
    main()













