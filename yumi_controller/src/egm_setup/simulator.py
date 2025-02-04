#!/usr/bin/env python3

# This is a super basic "simulator" or more like it integrates the velocity commands at 250 hz

import numpy as np

import rospy
import threading

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray as Float64MultiArrayMsg
from abb_egm_msgs.msg import EGMState, EGMChannelState
from abb_rapid_sm_addin_msgs.srv import SetSGCommand as SetSGCommandSrv, SetSGCommandRequest
from abb_robot_msgs.srv import TriggerWithResultCode as TriggerWithResultCodeSrv, TriggerWithResultCodeRequest
from abb_robot_msgs.msg import SystemState


class YumiJointState(object):
    def __init__(self,
        joint_pos=np.array([1.0, -2.0, -1.2, 0.6, -2.0, 1.0, 0.0, -1.0, -2.0, 1.2, 0.6, 2.0, 1.0, 0.0]),
        joint_vel=np.zeros(14),
        gripper_right_pos=np.zeros(2),
        gripper_left_pos=np.zeros(2),
        gripper_right_vel=np.zeros(2),
        gripper_left_vel=np.zeros(2)
    ):
        self.joint_pos = joint_pos  # only arm not gripper
        self.joint_vel = joint_vel  # only arm not gripper
        self.gripper_right_pos = gripper_right_pos
        self.gripper_left_pos = gripper_left_pos
        self.gripper_right_vel = gripper_right_vel
        self.gripper_left_vel = gripper_left_vel
    
    def velocity(self):
        return np.hstack([self.joint_vel,
                          self.gripper_right_vel,
                          self.gripper_left_vel])
    
    def position(self):
        return np.hstack([self.joint_pos,
                          self.gripper_right_pos,
                          self.gripper_left_pos])

    def update_pose(self, pose):
        self.joint_pos = pose[0:14]
        self.gripper_right_pos = pose[14:16]
        self.gripper_left_pos = pose[16:18]

    def update_velocity(self, vel):
        self.joint_vel = vel[0:14]
        self.gripper_right_vel = vel[14:16]
        self.gripper_left_vel = vel[16:18]


class Simulator(object):
    def __init__(self):
        self.update_rate = 500 #hz
        self.dt = 1/self.update_rate
        self.joint_state = YumiJointState(
            joint_pos=np.array([ 0.0, -2.270, -2.356, 0.524, 0.0, 0.670, 0.0,
                                     0.0, -2.270,  2.356, 0.524, 0.0, 0.670, 0.0]))

        self.lock = threading.Lock()

        arm_limit_upper = np.radians([ 168.5,   43.5,  168.5,     80,  290, 138,  229])
        arm_limit_lower = np.radians([-168.5, -143.5, -168.5, -123.5, -290, -88, -229])

        self.joint_pos_bound_upper = np.hstack([arm_limit_upper, arm_limit_upper, np.array([0.025, 0.025, 0.025, 0.025])])
        self.joint_pos_bound_lower = np.hstack([arm_limit_lower, arm_limit_lower, np.array([-0.0, -0.0, -0.0, -0.0])])
        self.target_gripper_pos = np.array([0.0, 0.0])
        # create ros service for grippers
        rospy.Service("/yumi/rws/sm_addin/set_sg_command", SetSGCommandSrv, self.receive_gripper_command)
        rospy.Service("/yumi/rws/sm_addin/run_sg_routine", TriggerWithResultCodeSrv, self.set_gripper_command)
        # name of gripper joints in urdf
        self.joint_names_grippers = ["gripper_l_joint", "gripper_l_joint_m", "gripper_r_joint", "gripper_r_joint_m"]
        self.gripper_position = np.array([0.0,0.0]) # used to store gripper commands until they are used

    def rapid_service(self, req: TriggerWithResultCodeRequest):
        print("started RAPID")
        return [1, ""]

    def callback(self, msg: Float64MultiArrayMsg):
        vel = np.asarray(msg.data)
        vel = np.hstack([vel[7:14], vel[0:7], np.zeros(4)])
        with self.lock:
            self.joint_state.update_velocity(vel)

    def update(self):
        # updates the pose
        with self.lock:
            pose = self.joint_state.position() + self.joint_state.velocity() * self.dt
            gripper_right = self.joint_state.position()[14:16]
            gripper_left = self.joint_state.position()[16:18]

        gripper_right_vel = (self.target_gripper_pos[0] - gripper_right) * self.dt
        pose[14:16] = gripper_right + gripper_right_vel
        gripper_left_vel = (self.target_gripper_pos[1] - gripper_left) * self.dt
        pose[16:18] = gripper_left + gripper_left_vel

        # hard joint limits 
        pose = np.clip(pose, self.joint_pos_bound_lower, self.joint_pos_bound_upper)
        self.joint_state.update_pose(pose)

    def receive_gripper_command(self, SetSGCommand: SetSGCommandRequest):
        # callback for gripper set_sg_command service, only 3 functionalities emulated, move to, grip in and grip out.
        # index for left gripper task
        if SetSGCommand.task == "T_ROB_R":
            index_a = 0
        # index for the right gripper
        elif SetSGCommand.task == "T_ROB_L":
            index_a = 1
        else:
            return [2, ""]  # returns failure state as service is finished

        if SetSGCommand.command == 5:  # move to
            self.gripper_position[index_a] = SetSGCommand.target_position/1000  # convert mm to meters

        elif SetSGCommand.command == 6:  # grip in
            self.gripper_position[index_a] = 0
        elif SetSGCommand.command == 7:  # grip out
            self.gripper_position[index_a] = 0.025
        else:
            return [2, ""]  # returns failure state as service is finished

        return [1, ""]  # returns success state as service is finished

    def set_gripper_command(self, SetSGCommand: SetSGCommandRequest):
        # callback for run_sg_routine, runs the gripper commands, i.e. grippers wont move before this service is called.
        self.target_gripper_pos = np.copy(self.gripper_position)
        return [1, ""]


def main():
    # starting ROS node and subscribers
    rospy.init_node("yumi_simulator", anonymous=True)
     
    pub_joint_states = rospy.Publisher("/yumi/egm/joint_states", JointState, queue_size=1)
    pub_egm_state = rospy.Publisher("/yumi/egm/egm_states", EGMState, queue_size=1)
    pub_rws_state = rospy.Publisher("/yumi/rws/system_states", SystemState, queue_size=1)

    simulator = Simulator()

    rospy.Subscriber("/yumi/egm/joint_group_velocity_controller/command", Float64MultiArrayMsg, simulator.callback, queue_size=1, tcp_nodelay=True)
    srv_start_rapid = rospy.Service("/yumi/rws/start_rapid", TriggerWithResultCodeSrv, simulator.rapid_service)

    msg = JointState(
        name = [
            "yumi_robr_joint_1", "yumi_robr_joint_2", "yumi_robr_joint_3", "yumi_robr_joint_4", "yumi_robr_joint_5", "yumi_robr_joint_6", "yumi_robr_joint_7",  
            "yumi_robl_joint_1", "yumi_robl_joint_2", "yumi_robl_joint_3", "yumi_robl_joint_4", "yumi_robl_joint_5", "yumi_robl_joint_6", "yumi_robl_joint_7", 
            "gripper_r_joint", "gripper_r_joint_m", "gripper_l_joint", "gripper_l_joint_m"])
    msg_egm_state = EGMState(
        egm_channels = [
            EGMChannelState(active=True), 
            EGMChannelState(active=True)])
    msg_rws_state = SystemState(
        motors_on = True,
        auto_mode = True,
        rapid_running = True)

    seq = 1
    rate = rospy.Rate(simulator.update_rate) 
    while not rospy.is_shutdown():
        simulator.update()
        
        msg.header.stamp = rospy.Time.now()
        msg.header.seq = seq
        msg.position = simulator.joint_state.position().tolist()
        msg.velocity = simulator.joint_state.velocity().tolist()
        
        pub_joint_states.publish(msg)
        pub_egm_state.publish(msg_egm_state)
        pub_rws_state.publish(msg_rws_state)
        
        rate.sleep()
        seq += 1
    
    srv_start_rapid.shutdown("simulator shutting down")


if __name__ == "__main__":
    main()
