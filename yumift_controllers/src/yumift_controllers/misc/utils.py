import rospy, rospkg
import numpy as np, quaternion as quat
import yaml
from pathlib import Path

from geometry_msgs.msg import (
    Point as PointMsg,
    PoseStamped as PoseStampedMsg, 
    Quaternion as QuaternionMsg, 
    Twist as TwistMsg,
)
from yumift_msgs.msg import RobotState as RobotStateMsg

from yumift_common.msg_utils import JacobianMsg_to_ndarray, PoseMsg_to_Frame, TwistMsg_to_ndarray, WrenchMsg_to_ndarray
from yumift_common.robot_state import YumiCoordinatedRobotState
from yumift_controllers.impl.trajectory import YumiParam

from dynamicals.utils import Frame, jacobian_combine


def load_config(filename : str):
    pkg_path = Path(rospkg.RosPack().get_path("yumift_controllers")) / "config" / filename
    with open(str(pkg_path)) as f:
        data = yaml.safe_load(f)
    return data


def sanitize_pos(pos: PointMsg, fallback : np.ndarray = np.zeros(3)) -> np.ndarray:
    return np.array([pos.x, pos.y, pos.z]) if pos else fallback

def sanitize_rot(rot: QuaternionMsg, fallback : np.quaternion = quat.one) -> np.quaternion:
    return quat.quaternion(rot.w, rot.x, rot.y, rot.z) if rot else fallback

def sanitize_vel(vel: TwistMsg, fallback : np.ndarray = np.zeros(6)) -> np.ndarray:
    return np.array([
        vel.linear.x, vel.linear.y, vel.linear.z, 
        vel.angular.x, vel.angular.y, vel.angular.z]) if vel else fallback

def quat_to_xyzw(q: np.quaternion) -> np.ndarray:
    return np.roll(quat.as_float_array(q), -1)


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
    yumi_state.effector_wrc = np.concatenate([pose_wrench_r, pose_wrench_l])
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
    # for the robot's flanges, a small workaround is needed, namely setting the 
    # explicit poses of the grippers. This is totally fine though because this 
    # transformation will only be used by the control law, which doesn't require 
    # the state to be sound nor complete (i.e. we use this object as a container
    # of values, no need to be consistant with every field in it)
    yumi_state.pose_gripper_r.pos = yumi_param.pose_right.pos
    yumi_state.pose_gripper_r.rot = yumi_param.pose_right.rot
    yumi_state.pose_gripper_r.vel = yumi_param.pose_right.vel
    yumi_state.pose_gripper_l.pos = yumi_param.pose_left.pos
    yumi_state.pose_gripper_l.rot = yumi_param.pose_left.rot
    yumi_state.pose_gripper_l.vel = yumi_param.pose_left.vel

    return yumi_state
