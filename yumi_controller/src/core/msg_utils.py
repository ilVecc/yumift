import numpy as np
import quaternion as quat

import rospy

from geometry_msgs.msg import Pose, PoseStamped, Wrench, Twist
from yumi_controller.msg import RobotState, Jacobian

from .robot_state import YumiCoordinatedRobotState
from .trajectory import YumiParam
from dynamics.utils import Frame, jacobian_combine


def TwistMsg_to_ndarray(twist: Twist):
    vx = twist.linear.x
    vy = twist.linear.y
    vz = twist.linear.z
    wx = twist.angular.x
    wy = twist.angular.y
    wz = twist.angular.z
    return np.array([vx, vy, vz, wx, wy, wz])


def WrenchMsg_to_ndarray(wrench: Wrench):
    fx = wrench.force.x
    fy = wrench.force.y
    fz = wrench.force.z
    mx = wrench.torque.x
    my = wrench.torque.y
    mz = wrench.torque.z
    return np.array([fx, fy, fz, mx, my, mz])


def PoseMsg_to_Frame(pose_msg: Pose):
    return Frame(
        position=np.array([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z]),
        quaternion=np.quaternion(pose_msg.orientation.w, pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z))


def Jacobian_to_ndarray(jacobian: Jacobian):
    jac = np.zeros((6, jacobian.dof), dtype=np.float)
    jac[0, :] = jacobian.vx
    jac[1, :] = jacobian.vy
    jac[2, :] = jacobian.vz
    jac[3, :] = jacobian.wx
    jac[4, :] = jacobian.wy
    jac[5, :] = jacobian.wz
    return jac


def YumiParam_to_YumiCoordinatedRobotState(yumi_param: YumiParam):
    """ Transforms a desired Yumi parameter into a Yumi state.
    """
    yumi_state = YumiCoordinatedRobotState(
        grip_r=yumi_param.grip_right, 
        grip_l=yumi_param.grip_left)
    # Since the `pose_*` arguments of `YumiCoordinatedRobotState` are the ones 
    # for the robot's flanges, a small workaround is needed. This is totally 
    # fine though because this transformation will only be used by the control 
    # law, which doesn't require the state to be sound nor complete
    yumi_state.pose_gripper_r = Frame(
        yumi_param.position[:3],
        yumi_param.rotation[0],
        yumi_param.velocity[:6])
    yumi_state.pose_gripper_l = Frame(
        yumi_param.position[3:],
        yumi_param.rotation[1],
        yumi_param.velocity[6:])
    return yumi_state


def Frame_to_PoseStamped(pose: Frame, parent: str = "yumi_base_link"):
    msg = PoseStamped()
    msg.header.frame_id = parent
    msg.header.stamp = rospy.Time.now()
    msg.pose.position.x = pose.pos[0]
    msg.pose.position.y = pose.pos[1]
    msg.pose.position.z = pose.pos[2]
    msg.pose.orientation.w = pose.rot.w
    msg.pose.orientation.x = pose.rot.x
    msg.pose.orientation.y = pose.rot.y
    msg.pose.orientation.z = pose.rot.z
    return msg


def RobotState_to_YumiCoordinatedRobotState(robot_state: RobotState):
    yumi_state = YumiCoordinatedRobotState()
    
    yumi_state.joint_pos = np.array(robot_state.jointState[0].position[:7] + robot_state.jointState[1].position[:7])
    yumi_state.joint_vel = np.array(robot_state.jointState[0].velocity[:7] + robot_state.jointState[1].velocity[:7])
    yumi_state.grip_r = robot_state.jointState[0].position[7]
    yumi_state.grip_l = robot_state.jointState[1].position[7]
    
    yumi_state.pose_gripper_r = PoseMsg_to_Frame(robot_state.pose[0])
    yumi_state.pose_gripper_r.vel = TwistMsg_to_ndarray(robot_state.poseTwist[0])
    yumi_state.pose_gripper_l = PoseMsg_to_Frame(robot_state.pose[1])
    yumi_state.pose_gripper_l.vel = TwistMsg_to_ndarray(robot_state.poseTwist[1])
    yumi_state.pose_wrench = np.concatenate([WrenchMsg_to_ndarray(robot_state.poseWrench[0]), WrenchMsg_to_ndarray(robot_state.poseWrench[1])])
    
    yumi_state.pose_abs = PoseMsg_to_Frame(robot_state.pose[2])
    yumi_state.pose_abs.vel = TwistMsg_to_ndarray(robot_state.poseTwist[2])
    yumi_state.pose_wrench_abs = WrenchMsg_to_ndarray(robot_state.poseWrench[2])
    
    yumi_state.pose_rel = PoseMsg_to_Frame(robot_state.pose[3])
    yumi_state.pose_rel.vel = TwistMsg_to_ndarray(robot_state.poseTwist[3])
    yumi_state.pose_wrench_rel = WrenchMsg_to_ndarray(robot_state.poseWrench[3])
    
    yumi_state.pose_elbow_r = PoseMsg_to_Frame(robot_state.pose[4])
    yumi_state.pose_elbow_r.vel = TwistMsg_to_ndarray(robot_state.poseTwist[4])
    
    yumi_state.pose_elbow_l = PoseMsg_to_Frame(robot_state.pose[5])
    yumi_state.pose_elbow_l.vel = TwistMsg_to_ndarray(robot_state.poseTwist[5])
    
    yumi_state.jacobian_grippers = jacobian_combine(Jacobian_to_ndarray(robot_state.jacobian[0]), Jacobian_to_ndarray(robot_state.jacobian[1]))
    yumi_state.jacobian_coordinated = np.vstack([Jacobian_to_ndarray(robot_state.jacobian[2]), Jacobian_to_ndarray(robot_state.jacobian[3])])
    yumi_state.jacobian_elbows = jacobian_combine(Jacobian_to_ndarray(robot_state.jacobian[4]), Jacobian_to_ndarray(robot_state.jacobian[5]))
    
    return yumi_state
