import numpy as np

from yumift_common.constants import YumiRobotConstants


q_avg = np.concatenate([(YumiRobotConstants.JOINT_POS_UB + YumiRobotConstants.JOINT_POS_LB) / 2, 
                        (YumiRobotConstants.JOINT_POS_UB + YumiRobotConstants.JOINT_POS_LB) / 2])
q_span = np.concatenate([YumiRobotConstants.JOINT_POS_UB - YumiRobotConstants.JOINT_POS_LB,
                         YumiRobotConstants.JOINT_POS_UB - YumiRobotConstants.JOINT_POS_LB])


def secondary_nothing(q, dq): 
    return np.zeros(YumiRobotConstants.DOF)

def secondary_neutral(q, dq, k : float = 10): 
    return - k * (1/YumiRobotConstants.DOF) * (q - YumiRobotConstants.JOINT_POS_NEUTRAL) / q_span ** 2

def secondary_center(q, dq, k : float = 10): 
    return - k * (1/YumiRobotConstants.DOF) * (q - q_avg) / q_span ** 2
