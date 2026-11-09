import numpy as np

from yumift_common.constants import YumiRobotConstants


q_avg = np.concatenate([(YumiRobotConstants.JOINT_POS_UB + YumiRobotConstants.JOINT_POS_LB) / 2, 
                        (YumiRobotConstants.JOINT_POS_UB + YumiRobotConstants.JOINT_POS_LB) / 2])
q_span = np.concatenate([np.abs(YumiRobotConstants.JOINT_POS_UB - YumiRobotConstants.JOINT_POS_LB),
                         np.abs(YumiRobotConstants.JOINT_POS_UB - YumiRobotConstants.JOINT_POS_LB)])

# YumiSizedArray = np.ndarray[YumiRobotConstants.DOF]
YumiSizedArray = np.ndarray

def secondary_nothing(q : YumiSizedArray, dq : YumiSizedArray) -> YumiSizedArray: 
    return np.zeros(YumiRobotConstants.DOF)

def secondary_configuration(q : YumiSizedArray, dq : YumiSizedArray, q0 : YumiSizedArray, k : float = 10):
    return - k * (1/YumiRobotConstants.DOF) * (q - q0) / q_span ** 2

def secondary_neutral(q : YumiSizedArray, dq : YumiSizedArray, k : float = 10) -> YumiSizedArray: 
    return secondary_configuration(q, dq, YumiRobotConstants.JOINT_POS_NEUTRAL, k)

def secondary_center(q : YumiSizedArray, dq : YumiSizedArray, k : float = 10) -> YumiSizedArray: 
    return secondary_configuration(q, dq, q_avg, k)

class SecondaryManipulability():
    
    def __init__(self) -> None:
        self.prevJ = None
    
    @staticmethod
    def manipulability(J : np.ndarray):
        return np.sqrt(np.linalg.det(J @ J.T))
    
    def dJ(self, J, dt):
        return (J - self.prevJ) / dt
    
    def __call__(self, J : np.ndarray, dt : float, q : np.ndarray, dq : np.ndarray) -> np.ndarray:
        Jpinv = np.linalg.pinv(J)
        # TODO what about dq = 0 ?
        ret = self.manipulability(J) * np.trace(self.dJ(J, dt) @ Jpinv) / dq
        self.prevJ = J
        return ret