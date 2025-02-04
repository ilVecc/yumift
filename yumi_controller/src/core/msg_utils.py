import numpy as np
import quaternion as quat

import rospy

from geometry_msgs.msg import Pose as PoseMsg, PoseStamped as PoseStampedMsg, Wrench as WrenchMsg, Twist as TwistMsg
from yumi_controller.msg import RobotState as RobotStateMsg, Jacobian as JacobianMsg

from .robot_state import YumiCoordinatedRobotState
from .trajectory import YumiParam
from dynamics.utils import Frame, jacobian_combine

### ROS messages to interal object representation

def TwistMsg_to_ndarray(twist: TwistMsg):
    vx = twist.linear.x
    vy = twist.linear.y
    vz = twist.linear.z
    wx = twist.angular.x
    wy = twist.angular.y
    wz = twist.angular.z
    return np.array([vx, vy, vz, wx, wy, wz])

def WrenchMsg_to_ndarray(wrench: WrenchMsg):
    fx = wrench.force.x
    fy = wrench.force.y
    fz = wrench.force.z
    mx = wrench.torque.x
    my = wrench.torque.y
    mz = wrench.torque.z
    return np.array([fx, fy, fz, mx, my, mz])

def PoseMsg_to_Frame(pose_msg: PoseMsg, frame: Frame = None):
    pos = np.array([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z])
    rot = np.quaternion(pose_msg.orientation.w, pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z)
    if frame is None:
        return Frame(pos, rot)
    else:
        frame.pos = pos
        frame.rot = rot
        return frame

def JacobianMsg_to_ndarray(jacobian: JacobianMsg, array: np.ndarray = None):
    if array is None:
        jac = np.zeros((6, jacobian.dof), dtype=np.float)
    jac[0, :] = jacobian.vx
    jac[1, :] = jacobian.vy
    jac[2, :] = jacobian.vz
    jac[3, :] = jacobian.wx
    jac[4, :] = jacobian.wy
    jac[5, :] = jacobian.wz
    return jac

def RobotStateMsg_to_YumiCoordinatedRobotState(robot_state: RobotStateMsg, yumi_state = YumiCoordinatedRobotState()):
    yumi_state.joint_pos = np.array(robot_state.jointState[0].position[:7] + robot_state.jointState[1].position[:7])
    yumi_state.joint_vel = np.array(robot_state.jointState[0].velocity[:7] + robot_state.jointState[1].velocity[:7])
    yumi_state.grip_r = robot_state.jointState[0].position[7]
    yumi_state.grip_l = robot_state.jointState[1].position[7]
    
    if "gripper_r" in robot_state.poseName:
        yumi_state.pose_gripper_r = PoseMsg_to_Frame(robot_state.pose[0])
        yumi_state.pose_gripper_r.vel = TwistMsg_to_ndarray(robot_state.poseTwist[0])
        pose_wrench_r = WrenchMsg_to_ndarray(robot_state.poseWrench[0])
        jac_gripper_r = JacobianMsg_to_ndarray(robot_state.jacobian[0])
    else:
        pose_wrench_r = np.zeros(6)
        jac_gripper_r = np.zeros((6,7))
    if "gripper_l" in robot_state.poseName:
        yumi_state.pose_gripper_l = PoseMsg_to_Frame(robot_state.pose[1])
        yumi_state.pose_gripper_l.vel = TwistMsg_to_ndarray(robot_state.poseTwist[1])
        pose_wrench_l = WrenchMsg_to_ndarray(robot_state.poseWrench[1])
        jac_gripper_l = JacobianMsg_to_ndarray(robot_state.jacobian[1])
    else:
        pose_wrench_l = np.zeros(6)
        jac_gripper_l = np.zeros((6,7))
    yumi_state.effector_wrench = np.concatenate([pose_wrench_r, pose_wrench_l])
    yumi_state.jacobian_grippers = jacobian_combine(jac_gripper_r, jac_gripper_l)
    
    if "absolute" in robot_state.poseName:
        yumi_state.pose_abs = PoseMsg_to_Frame(robot_state.pose[2], yumi_state.pose_abs)
        yumi_state.pose_abs.vel = TwistMsg_to_ndarray(robot_state.poseTwist[2])
        yumi_state.pose_wrench_abs = WrenchMsg_to_ndarray(robot_state.poseWrench[2])
        jac_abs = JacobianMsg_to_ndarray(robot_state.jacobian[2])
    else:
        yumi_state.pose_wrench_abs = np.zeros(6)
        jac_abs = np.zeros((6,14))
    if "relative" in robot_state.poseName:
        yumi_state.pose_rel = PoseMsg_to_Frame(robot_state.pose[3], yumi_state.pose_rel)
        yumi_state.pose_rel.vel = TwistMsg_to_ndarray(robot_state.poseTwist[3])
        yumi_state.pose_wrench_rel = WrenchMsg_to_ndarray(robot_state.poseWrench[3])
        jac_rel = JacobianMsg_to_ndarray(robot_state.jacobian[3])
    else:
        yumi_state.pose_wrench_rel = np.zeros(6)
        jac_rel = np.zeros((6,14))
    yumi_state.jacobian_coordinated = np.vstack([jac_abs, jac_rel])
    
    if "elbow_r" in robot_state.poseName:
        yumi_state.pose_elbow_r = PoseMsg_to_Frame(robot_state.pose[4])
        yumi_state.pose_elbow_r.vel = TwistMsg_to_ndarray(robot_state.poseTwist[4])
        jac_elb_r = JacobianMsg_to_ndarray(robot_state.jacobian[4])
    else:
        jac_elb_r = np.zeros((6,7))
    if "elbow_l" in robot_state.poseName:
        yumi_state.pose_elbow_l = PoseMsg_to_Frame(robot_state.pose[5])
        yumi_state.pose_elbow_l.vel = TwistMsg_to_ndarray(robot_state.poseTwist[5])
        jac_elb_l = JacobianMsg_to_ndarray(robot_state.jacobian[5])
    else:
        jac_elb_l = np.zeros((6,7))
    yumi_state.jacobian_elbows = jacobian_combine(jac_elb_r, jac_elb_l)
    
    return yumi_state

### interal object representation to ROS messages

def Frame_to_PoseStampedMsg(pose: Frame, parent: str = "yumi_base_link"):
    msg = PoseStampedMsg()
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

### interal object representation convertion

def YumiParam_to_YumiCoordinatedRobotState(yumi_param: YumiParam, yumi_state = YumiCoordinatedRobotState()):
    """ Transforms a desired Yumi parameter into a Yumi state.
    """
    yumi_state.grip_r=yumi_param.grip_right
    yumi_state.grip_l=yumi_param.grip_left
        
    # Since the `pose_*` arguments of `YumiCoordinatedRobotState` are the ones 
    # for the robot's flanges, a small workaround is needed. This is totally 
    # fine though because this transformation will only be used by the control 
    # law, which doesn't require the state to be sound nor complete
    yumi_state.pose_gripper_r.pos = yumi_param.position[:3]
    yumi_state.pose_gripper_r.rot = yumi_param.rotation[0]
    yumi_state.pose_gripper_r.vel = yumi_param.velocity[:6]
    yumi_state.pose_gripper_l.pos = yumi_param.position[3:]
    yumi_state.pose_gripper_l.rot = yumi_param.rotation[1]
    yumi_state.pose_gripper_l.vel = yumi_param.velocity[6:]
    
    return yumi_state
