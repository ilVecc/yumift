import numpy as np
import quaternion as quat

from dynamics.utils import RobotState, Frame, jacobian_combine


class YumiRobotState(RobotState):
    """ State of the robot [right, left]
    """
    def __init__(
        self,
        joint_pos: np.ndarray = np.zeros(14),
        joint_vel: np.ndarray = np.zeros(14),
        joint_torque: np.ndarray = np.zeros(14),
        pose_pos: np.ndarray = np.zeros(6),
        pose_rot: np.ndarray = np.stack([quat.one, quat.one]),
        pose_vel: np.ndarray = np.zeros(12),
        pose_wrench: np.ndarray = np.zeros(12),
        jacobian: np.ndarray = np.zeros((12, 14)),
        grip_r: float = 0.,
        grip_l: float = 0.
    ):
        super().__init__(14)  # ignore all variables, instead wrap _right and _left using properties
        self._right = RobotState(7, joint_pos[:7], joint_vel[:7], None, joint_torque[:7], 
                                    pose_pos[:3], pose_rot[0], pose_vel[:6], None, pose_wrench[:6], 
                                    jacobian[0:6, 0:7])
        self._left = RobotState(7, joint_pos[7:], joint_vel[7:], None, joint_torque[7:], 
                                   pose_pos[3:], pose_rot[1], pose_vel[6:], None, pose_wrench[6:], 
                                   jacobian[6:12, 7:14])
        # gripping
        self.grip_r = grip_r
        self.grip_l = grip_l

    ####### RIGHT #######
    @property
    def joint_pos_r(self):
        return self._right.joint_pos
    
    @property
    def joint_vel_r(self):
        return self._right.joint_vel
    
    @property
    def joint_acc_r(self):
        return self._right.joint_acc
    
    @property
    def joint_torque_r(self):
        return self._right.joint_tau

    @property
    def pose_pos_r(self):
        return self._right.effector_pos

    @property
    def pose_rot_r(self):
        return self._right.effector_rot
    
    @property
    def pose_vel_r(self):
        return self._right.effector_vel
    
    @property
    def pose_vel_lin_r(self):
        return self._right.effector_vel_lin

    @property
    def pose_vel_ang_r(self):
        return self._right.effector_vel_ang
    
    @property
    def pose_acc_r(self):
        return self._right.effector_acc
    
    @property
    def pose_acc_lin_r(self):
        return self._right.effector_acc_lin

    @property
    def pose_acc_ang_r(self):
        return self._right.effector_acc_ang
    
    @property
    def pose_wrench_r(self):
        return self._right.effector_wrench
    
    @property
    def pose_force_r(self):
        return self._right.effector_force
    
    @property
    def pose_torque_r(self):
        return self._right.effector_moment
    
    @property
    def jacobian_r(self):
        return self._right.jacobian
    
    ####### LEFT #######
    @property
    def joint_pos_l(self):
        return self._left.joint_pos
    
    @property
    def joint_vel_l(self):
        return self._left.joint_vel
    
    @property
    def joint_acc_l(self):
        return self._left.joint_acc
    
    @property
    def joint_torque_l(self):
        return self._left.joint_tau

    @property
    def pose_pos_l(self):
        return self._left.effector_pos

    @property
    def pose_rot_l(self):
        return self._left.effector_rot
    
    @property
    def pose_vel_l(self):
        return self._left.effector_vel
    
    @property
    def pose_vel_lin_l(self):
        return self._left.effector_vel_lin

    @property
    def pose_vel_ang_l(self):
        return self._left.effector_vel_ang
    
    @property
    def pose_acc_l(self):
        return self._left.effector_acc
    
    @property
    def pose_acc_lin_l(self):
        return self._left.effector_acc_lin

    @property
    def pose_acc_ang_l(self):
        return self._left.effector_acc_ang
    
    @property
    def pose_wrench_l(self):
        return self._left.effector_wrench
    
    @property
    def pose_force_l(self):
        return self._left.effector_force
    
    @property
    def pose_torque_l(self):
        return self._left.effector_moment
    
    @property
    def jacobian_l(self):
        return self._left.jacobian

    ####### OVERALL #######
    @property
    def joint_pos(self):
        return np.concatenate([self.joint_pos_r, self.joint_pos_l])
    
    # TODO refactor this
    @joint_pos.setter
    def joint_pos(self, value):
        self._right._joint_pos = value[:7]
        self._left._joint_pos = value[7:]
    
    @property
    def joint_vel(self):
        return np.concatenate([self.joint_vel_r, self.joint_vel_l])
    
    # TODO refactor this
    @joint_vel.setter
    def joint_vel(self, value):
        self._right._joint_vel = value[:7]
        self._left._joint_vel = value[7:]
        
    @property
    def joint_acc(self):
        return np.concatenate([self.joint_acc_r, self.joint_acc_l])
    
    @property
    def joint_tau(self):
        return np.concatenate([self.joint_torque_r, self.joint_torque_l])

    # TODO refactor this
    @joint_tau.setter
    def joint_tau(self, value):
        self._right._joint_tau = value[:7]
        self._left._joint_tau = value[7:]

    @property
    def effector_pos(self):
        return np.concatenate([self.pose_pos_r, self.pose_pos_l])

    @property
    def effector_rot(self):
        return np.stack([self.pose_rot_r, self.pose_rot_l])
    
    @property
    def effector_vel(self):
        return np.concatenate([self.pose_vel_r, self.pose_vel_l])
        
    @property
    def effector_vel_lin(self):
        return np.concatenate([self.pose_vel_lin_r, self.pose_vel_lin_l])

    @property
    def effector_vel_ang(self):
        return np.concatenate([self.pose_vel_ang_r, self.pose_vel_ang_l])
    
    @property
    def effector_acc(self):
        return np.concatenate([self.pose_acc_r, self.pose_acc_l])
    
    @property
    def effector_acc_lin(self):
        return np.concatenate([self.pose_acc_lin_r, self.pose_acc_lin_l])

    @property
    def effector_acc_ang(self):
        return np.concatenate([self.pose_acc_ang_r, self.pose_acc_ang_l])
    
    @property
    def effector_wrench(self):
        return np.concatenate([self.pose_wrench_r, self.pose_wrench_l])
    
    # TODO refactor this
    @effector_wrench.setter
    def effector_wrench(self, value):
        self._right._effector_wrc = value[:6]
        self._left._effector_wrc = value[6:]
    
    @property
    def effector_force(self):
        return np.concatenate([self.pose_force_r, self.pose_force_l])
    
    @property
    def effector_moment(self):
        return np.concatenate([self.pose_torque_r, self.pose_torque_l])
    
    @property
    def jacobian(self):
        return jacobian_combine(self.jacobian_r, self.jacobian_l)
    
    # TODO refactor this
    @jacobian.setter
    def jacobian(self, value):
        self._right._jac = value[:6, :7]
        self._left._jac = value[6:, 7:]


class YumiCoordinatedRobotState(YumiRobotState):
    """ State of the robot [right, left].
        Symmetry [-1,+1] indicates the balance between right (+1) and left (-1).
    """
    def __init__(
        self,
        joint_pos: np.ndarray = np.zeros(14),
        joint_vel: np.ndarray = np.zeros(14),
        joint_torque: np.ndarray = np.zeros(14),
        pose_pos: np.ndarray = np.zeros(6),
        pose_rot: np.ndarray = np.stack([quat.one, quat.one]),
        pose_vel: np.ndarray = np.zeros(12),
        pose_wrench: np.ndarray = np.zeros(12),
        jacobian: np.ndarray = np.zeros((12, 14)),
        grip_r: float = 0.,
        grip_l: float = 0.,
        symmetry: float = 0.
    ):
        super().__init__(joint_pos, joint_vel, joint_torque, pose_pos, pose_rot, pose_vel, pose_wrench, jacobian, grip_r, grip_l)
        # grippers
        self.pose_gripper_r = Frame()
        self.pose_gripper_l = Frame()
        self.jacobian_grippers: np.ndarray = np.zeros((12, 14))
        # elbows
        self.pose_elbow_r = Frame()
        self.pose_elbow_l = Frame()
        self.jacobian_elbows: np.ndarray = np.zeros((12, 8))
        # coordinated
        assert -1 < symmetry and symmetry < +1, "Symmetry value must be in range [-1,+1]"
        self.alpha = (symmetry + 1) * 0.5
        self.pose_abs = Frame()
        self.pose_rel = Frame()
        self.pose_wrench_abs: np.ndarray = np.zeros(6)
        self.pose_wrench_rel: np.ndarray = np.zeros(6)
        self.jacobian_coordinated: np.ndarray = np.zeros((12, 14))
    
    @property
    def poses_individual(self):
        return self.pose_gripper_r, self.pose_gripper_l
    
    @property
    def poses_coordinated(self):
        return self.pose_abs, self.pose_rel
        
    @property
    def jacobian_gripper_r(self):
        return self.jacobian_grippers[:6, :7]

    @property
    def jacobian_gripper_l(self):
        return self.jacobian_grippers[6:, 7:]
    
    @property
    def jacobian_elbow_r(self):
        return self.jacobian_elbows[:6, :4]

    @property
    def jacobian_elbow_l(self):
        return self.jacobian_elbows[6:, 4:]
    
    @property
    def jacobian_coordinated_abs(self):
        return self.jacobian_coordinated[:6, :]

    @property
    def jacobian_coordinated_rel(self):
        return self.jacobian_coordinated[6:, :]
