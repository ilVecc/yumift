#!/usr/bin/env python3
import rospy

import numpy as np
from threading import Lock

from std_msgs.msg import Float64MultiArray as Float64MultiArrayMsg
from sensor_msgs.msg import JointState
from abb_egm_msgs.msg import EGMState, EGMChannelState
from abb_robot_msgs.msg import SystemState
from abb_robot_msgs.srv import TriggerWithResultCode as TriggerWithResultCodeSrv, TriggerWithResultCodeRequest
from abb_rapid_sm_addin_msgs.srv import SetSGCommand as SetSGCommandSrv, SetSGCommandRequest


class YumiJointState(object):
    def __init__(self,
        joint_pos=np.array([ 1.0, -2.0, -1.2, 0.6, -2.0, 1.0, 0.0, 
                            -1.0, -2.0,  1.2, 0.6,  2.0, 1.0, 0.0]),
        joint_vel=np.zeros(14),
        grip_pos=np.zeros(4),
        grip_vel=np.zeros(4)
    ):
        self.joint_grip_pos = np.hstack([joint_pos, grip_pos])  # arm + gripper not gripper
        self.joint_grip_vel = np.hstack([joint_vel, grip_vel])  # arm + gripper not gripper
    
    def pos(self):
        return self.joint_grip_pos
    
    def arm_pos(self):
        return self.joint_grip_pos[0:14]
    
    def grip_pos(self):
        return self.joint_grip_pos[14:18]

    def vel(self):
        return self.joint_grip_vel

    def arm_vel(self):
        return self.joint_grip_vel[0:14]

    def grip_vel(self):
        return self.joint_grip_vel[14:18]
    
    def update_pos(self, pos):
        self.joint_grip_pos = pos

    def update_arm_vel(self, vel):
        self.joint_grip_vel[0:14] = vel

    def update_grip_vel(self, vel):
        self.joint_grip_vel[14:18] = vel


class Emulator(object):
    """ This is a simple kinematics simulator which integrates the input 
        velocity commands at given rate
    """
    def __init__(self, update_rate : int = 500):
        self.update_rate = update_rate
        self.dt = 1/self.update_rate
        
        self.lock = Lock()
        self.joint_state = YumiJointState(np.array([ 0.0, -2.270, -2.356, 0.524, 0.0, 0.670, 0.0,
                                                     0.0, -2.270,  2.356, 0.524, 0.0, 0.670, 0.0]))
        # placeholder variable for gripper command until it's consumed by `run_sg_routine` service
        self._cache_grip_pos_command = np.zeros(2)
        self.target_gripper_pos = np.zeros(4)
        
        arm_limit_upper = np.radians([ 168.5,   43.5,  168.5,     80,  290, 138,  229])
        arm_limit_lower = np.radians([-168.5, -143.5, -168.5, -123.5, -290, -88, -229])
        self.joint_pos_bound_upper = np.hstack([arm_limit_upper, arm_limit_upper, np.array([0.025, 0.025, 0.025, 0.025])])
        self.joint_pos_bound_lower = np.hstack([arm_limit_lower, arm_limit_lower, np.array([-0.0, -0.0, -0.0, -0.0])])
        
        # simulate EGM/RWS status topics
        self._cache_jointstate_msg = JointState(name = [
            "yumi_robr_joint_1", "yumi_robr_joint_2", "yumi_robr_joint_3", "yumi_robr_joint_4", "yumi_robr_joint_5", "yumi_robr_joint_6", "yumi_robr_joint_7",  
            "yumi_robl_joint_1", "yumi_robl_joint_2", "yumi_robl_joint_3", "yumi_robl_joint_4", "yumi_robl_joint_5", "yumi_robl_joint_6", "yumi_robl_joint_7", 
            "gripper_r_joint", "gripper_r_joint_m", "gripper_l_joint", "gripper_l_joint_m"])
        self.pub_jointstates = rospy.Publisher("/yumi/egm/joint_states", JointState, queue_size=1)
        #
        self._cache_egm_state_msg = EGMState(egm_channels=[EGMChannelState(active=True), EGMChannelState(active=True)])
        self.pub_egm_states = rospy.Publisher("/yumi/egm/egm_states", EGMState, queue_size=1)
        #
        self._cache_rws_state_msg = SystemState(motors_on=True, auto_mode=True, rapid_running=True)
        self.pub_rws_states = rospy.Publisher("/yumi/rws/system_states", SystemState, queue_size=1)
        
        # create ROS service for grippers
        rospy.Service("/yumi/rws/sm_addin/set_sg_command", SetSGCommandSrv, self._srv_get_gripper_command)
        rospy.Service("/yumi/rws/sm_addin/run_sg_routine", TriggerWithResultCodeSrv, self._srv_set_gripper_command)
        
        # mock "start RAPID" service 
        srv_start_rapid = rospy.Service("/yumi/rws/start_rapid", TriggerWithResultCodeSrv, self._srv_start_rapid)
        rospy.on_shutdown(lambda : srv_start_rapid.shutdown("simulator shutting down"))
        
        # subscribe to joint velocity commands
        rospy.Subscriber("/yumi/egm/joint_group_velocity_controller/command", Float64MultiArrayMsg, self._callback_joint_command, queue_size=1)
        
        rospy.loginfo("Emulation for Yumi running")

    def _callback_joint_command(self, msg: Float64MultiArrayMsg):
        vel = np.asarray(msg.data[7:14] + msg.data[0:7])  # flip left and right
        with self.lock:
            self.joint_state.update_arm_vel(vel)

    def update(self):
        # updates the pose
        with self.lock:
            pos = self.joint_state.pos() + self.joint_state.vel() * self.dt
            grip_pos = self.joint_state.grip_pos()
            
            # hard joint limits
            pos = np.clip(pos, self.joint_pos_bound_lower, self.joint_pos_bound_upper)
            self.joint_state.update_pos(pos)
            
            # update grip velocity
            grip_vel = self.target_gripper_pos - grip_pos
            self.joint_state.update_grip_vel(grip_vel)
            
            self._cache_jointstate_msg.header.stamp = rospy.Time.now()
            self._cache_jointstate_msg.header.seq += 1
            self._cache_jointstate_msg.position = self.joint_state.pos().tolist()
            self._cache_jointstate_msg.velocity = self.joint_state.vel().tolist()
        
        self.pub_jointstates.publish(self._cache_jointstate_msg)
        self.pub_egm_states.publish(self._cache_egm_state_msg)
        self.pub_rws_states.publish(self._cache_rws_state_msg)

    def _srv_get_gripper_command(self, SetSGCommand: SetSGCommandRequest):
        # callback for gripper `set_sg_command` service, only 3 functionalities 
        # emulated: move_to, grip_in and grip_out.
        if SetSGCommand.task == "T_ROB_R":
            idx = 0
        elif SetSGCommand.task == "T_ROB_L":
            idx = 1
        else:
            return [2, ""]  # returns failure state as service is finished

        if SetSGCommand.command == 5:  # move_to
            self._cache_grip_pos_command[idx] = SetSGCommand.target_position * 0.001  # convert mm to meters
        elif SetSGCommand.command == 6:  # grip_in
            self._cache_grip_pos_command[idx] = 0
        elif SetSGCommand.command == 7:  # grip_out
            self._cache_grip_pos_command[idx] = 0.025
        else:
            return [2, ""]  # returns failure state as service is finished

        return [1, ""]  # returns success state as service is finished

    def _srv_set_gripper_command(self, SetSGCommand: SetSGCommandRequest):
        # callback for `run_sg_routine`, runs the gripper commands, 
        # i.e. grippers wont move before this service is called.
        self.target_gripper_pos[0] = self._cache_grip_pos_command[0]/2
        self.target_gripper_pos[1] = self._cache_grip_pos_command[0]/2
        self.target_gripper_pos[2] = self._cache_grip_pos_command[1]/2
        self.target_gripper_pos[3] = self._cache_grip_pos_command[1]/2
        return [1, ""]

    def _srv_start_rapid(self, req: TriggerWithResultCodeRequest):
        rospy.loginfo("started RAPID")
        return [1, ""]


if __name__ == "__main__":
    rospy.init_node("yumi_emulator", anonymous=True)
    
    emulator = Emulator(500)  #hz
    
    rate = rospy.Rate(emulator.update_rate) 
    while not rospy.is_shutdown():
        emulator.update()
        rate.sleep()
