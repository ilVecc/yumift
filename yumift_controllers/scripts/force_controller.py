#!/usr/bin/env python3
from typing_extensions import override

import rospy
import numpy as np

from yumift_controllers.common.device import YumiDevice, YumiDualDeviceState
from yumift_controllers.common.controller_base import MixedVelocityYumiAction, YumiDualController, YumiDualDeviceCommand
from yumift_controllers.common.parameters import ControllerParameters
from yumift_controllers.ik.algorithms import PINVIKAlgorithm

from dynamicals.systems import LPFilter
from yumift_controllers.ik.pinv_tasks import secondary_neutral


class YumiForceController(YumiDualController):
    def __init__(self):
        super().__init__(
            yumi_device=YumiDevice(coordinated_balance=0.5), 
            ikalgorithms=[PINVIKAlgorithm()])
        
        # position action
        self.des_baseP_r = np.array([0.40, -0.25, 0.1])
        self.des_baseP_l = np.array([0.40, +0.25, 0.1])
        self.Kp = np.eye(3) * 0.5
        # force action
        self.des_gripF_r = np.array([0, 0, 0])
        self.des_gripF_l = np.array([0, 0, 20])
        self.Kf = np.ones(3) * 0.005
        self.int_err_gripF_r = np.zeros(3)
        self.int_err_gripF_l = np.zeros(3)
        self.Ki = np.ones(3) * 0.001
        self.prev_err_gripF_r = np.zeros(3)
        self.prev_err_gripF_l = np.zeros(3)
        self.Kd = np.ones(3) * 0.0001
        self.filter_r = LPFilter(freq=50, n=3, h=ControllerParameters.dt)
        self.filter_l = LPFilter(freq=50, n=3, h=ControllerParameters.dt)
        # surface normal
        self.n = np.array([[0, 0, 1]]).T
        self.N = self.n @ self.n.T
        self.Q = np.eye(3) - self.N
        
        self.trajectory = lambda t : self.des_baseP_l + min(t,10)/10 * np.array([0, +0.10, 0])
        self.traj_time = rospy.Time.now()

    @override
    def reset(self, state: YumiDualDeviceState):
        """ Initialize the controller setting the current point as desired trajectory. 
            This method is called automatically every time EGM reconnects or after a 
            routine is completed.
        """
        # wait for Yumi
        while not self.device_is_ready():
            rospy.logwarn("Controller cannot be reset (Yumi is not ready, retrying in 5 seconds)")
            rospy.sleep(5)
    
    @override
    def policy(self, state: YumiDualDeviceState) -> MixedVelocityYumiAction:
        """ Calculate target velocity for the current time step.
        """
        # calculate timing information
        ctrl_dt = ControllerParameters.dt  # self.dt(state.time)
        
        # current position on the trajectory
        curr_baseP_r = state.pose_gripper_r.pos
        curr_baseP_l = state.pose_gripper_l.pos
        # exerted force on the object
        curr_gripF_r = -state.pose_gripper_r.reactTo(state.pose_wrench_r)[:3]
        curr_gripF_l = -state.pose_gripper_l.reactTo(state.pose_wrench_l)[:3]
        
        err_gripF_r = curr_gripF_r - self.des_gripF_r
        err_gripF_l = curr_gripF_l - self.des_gripF_l
        
        # P-action force
        tgt_gripV_r = -self.Kf * err_gripF_r
        tgt_gripV_l = -self.Kf * err_gripF_l
        # I-action force
        tgt_gripV_r += -self.Ki * self.filter_r.compute(err_gripF_r) # self.int_err_gripF_r
        tgt_gripV_l += -self.Ki * self.filter_l.compute(err_gripF_l) # self.int_err_gripF_l
        # # D-action force
        # tgt_gripV_r += -self.Kd @ self.filter_r.compute((err_gripF_r - self.prev_err_gripF_r) / ctrl_dt)
        # tgt_gripV_l += -self.Kd @ self.filter_l.compute((err_gripF_l - self.prev_err_gripF_l) / ctrl_dt)
        # P-action position
        # tgt_gripV_r += -self.Kp @ (curr_baseP_r - self.des_baseP_r)
        # tgt_gripV_l += -self.Kp @ (curr_baseP_l - self.des_baseP_l)
        
        # t = self.dt(self.time)
        # tgt_gripV_r = -self.Kf @ err_gripF_r - self.Kp @ (curr_baseP_r - self.des_baseP_r)
        # tgt_gripV_l = -self.Kf @ err_gripF_l - self.Kp @ (curr_baseP_l - self.trajectory(t))
        
        # tgt_gripV_r = -self.Kf * self.N @ (curr_gripF_r - self.des_gripF_r) + \
        #               -self.Q @ self.Kp @ self.Q @ (curr_gripP_r - self.des_gripP_r)
        # tgt_gripV_l = -self.Kf * self.N @ (curr_gripF_l - self.des_gripF_l) + \
        #               -self.Q @ self.Kp @ self.Q @ (curr_gripP_l - self.des_gripP_l)
        
        # store
        self.prev_err_gripF_r = err_gripF_r
        self.prev_err_gripF_l = err_gripF_l
        self.int_err_gripF_r += err_gripF_r * ctrl_dt
        self.int_err_gripF_l += err_gripF_l * ctrl_dt
        
        # CALCULATE TWISTS
        try:
            # set twists
            tgt_gripT_r, tgt_gripT_l = np.zeros(6), np.zeros(6)
            tgt_gripT_r[:3], tgt_gripT_l[:3] = tgt_gripV_r, tgt_gripV_l
            
            tgt_baseT_r = state.pose_gripper_r.actOn(tgt_gripT_r)
            tgt_baseT_l = state.pose_gripper_l.actOn(tgt_gripT_l)
            
            # get space based on control mode ...
            action = MixedVelocityYumiAction()
            action.control_space(MixedVelocityYumiAction.ControlSpace.INDIVIDUAL)
            action.timestep(ctrl_dt)
            action.velocity_right(tgt_baseT_r) 
            action.velocity_left(tgt_baseT_l)
            
        except Exception as ex:
            rospy.logfatal(f"Could not compute action (exception: {ex})")
            rospy.logfatal("Manually invoking fallback policy")
            action = self.fallback(state)
        
        return action

    @override
    def solve_action(self, state: YumiDualDeviceState, action: MixedVelocityYumiAction) -> YumiDualDeviceCommand:
        xdot = np.zeros(12)
        xdot[0:6] = action.get("velocity_right")
        xdot[6:12] = action.get("velocity_left")
        jacobian = state.jacobian_grippers
        
        joint_pos = np.zeros(14)
        joint_pos[0:7] = state.joint_pos_r
        joint_pos[7:14] = state.joint_pos_l
        
        jacobian_pinv = np.linalg.pinv(jacobian)
        ortho_proj = np.eye(14) - jacobian_pinv @ jacobian
        dq_target = jacobian_pinv @ xdot + ortho_proj @ secondary_neutral(joint_pos, None)

        return YumiDualDeviceCommand(dq_target)


if __name__ == "__main__":
    rospy.init_node("force_controller", anonymous=False)
    
    yumi_controller = YumiForceController()
    yumi_controller.ready()
    yumi_controller.start()  # locking
