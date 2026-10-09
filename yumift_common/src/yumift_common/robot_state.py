import numpy as np
import quaternion as quat

from dynamicals.utils import jacobian_combine
from dynamicals.utils import Frame
from dynamicals.common.robotics import RobotState


class YumiRobotState(RobotState):
    """ State of the robot [right, left] in base frame
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
        self._right = RobotState(7, joint_pos[:7], joint_vel[:7], None, joint_torque[:7], 
                                    pose_pos[:3], pose_rot[0], pose_vel[:6], None, pose_wrench[:6], 
                                    jacobian[0:6, 0:7], None)
        self._left = RobotState(7, joint_pos[7:], joint_vel[7:], None, joint_torque[7:], 
                                   pose_pos[3:], pose_rot[1], pose_vel[6:], None, pose_wrench[6:], 
                                   jacobian[6:12, 7:14], None)
        # super().__init__(14)  # ignore all variables, instead wrap _right and _left using properties
        self.dofs = 14
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
        return self._right.effector_wrc
    
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
        return self._left.effector_wrc
    
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
        qpos = np.zeros(14)
        qpos[0:7] = self.joint_pos_r
        qpos[7:14] = self.joint_pos_l
        return qpos
    
    # TODO refactor this
    @joint_pos.setter
    def joint_pos(self, value):
        self._right.joint_pos = value[:7]
        self._left.joint_pos = value[7:]
    
    @property
    def joint_vel(self):
        qvel = np.zeros(14)
        qvel[0:7] = self.joint_vel_r
        qvel[7:14] = self.joint_vel_l
        return qvel
    
    # TODO refactor this
    @joint_vel.setter
    def joint_vel(self, value):
        self._right.joint_vel = value[:7]
        self._left.joint_vel = value[7:]
        
    @property
    def joint_acc(self):
        qacc = np.zeros(14)
        qacc[0:7] = self.joint_acc_r
        qacc[7:14] = self.joint_acc_l
        return qacc
    
    @joint_acc.setter
    def joint_acc(self, value):
        self._right.joint_acc = value[:7]
        self._left.joint_acc = value[7:]
    
    @property
    def joint_tau(self):
        qtau = np.zeros(14)
        qtau[0:7] = self.joint_torque_r
        qtau[7:14] = self.joint_torque_l
        return qtau

    # TODO refactor this
    @joint_tau.setter
    def joint_tau(self, value):
        self._right.joint_tau = value[:7]
        self._left.joint_tau = value[7:]

    @property
    def effector_pos(self):
        xpos = np.zeros(6)
        xpos[0:3] = self.pose_pos_r
        xpos[3:6] = self.pose_pos_l
        return xpos

    @property
    def effector_rot(self):
        return np.stack([self.pose_rot_r, self.pose_rot_l])
    
    @property
    def effector_vel(self):
        xvel = np.zeros(12)
        xvel[0:6] = self.pose_vel_r
        xvel[6:12] = self.pose_vel_l
        return xvel
        
    @property
    def effector_vel_lin(self):
        xvel_lin = np.zeros(6)
        xvel_lin[0:3] = self.pose_vel_lin_r
        xvel_lin[3:6] = self.pose_vel_lin_l
        return xvel_lin

    @property
    def effector_vel_ang(self):
        xvel_ang = np.zeros(6)
        xvel_ang[0:3] = self.pose_vel_ang_r
        xvel_ang[3:6] = self.pose_vel_ang_l
        return xvel_ang
    
    @property
    def effector_acc(self):
        xacc = np.zeros(12)
        xacc[0:6] = self.pose_acc_r
        xacc[6:12] = self.pose_acc_l
        return xacc
    
    @property
    def effector_acc_lin(self):
        xacc_lin = np.zeros(6)
        xacc_lin[0:3] = self.pose_acc_lin_r
        xacc_lin[3:6] = self.pose_acc_lin_l
        return xacc_lin

    @property
    def effector_acc_ang(self):
        xacc_ang = np.zeros(6)
        xacc_ang[0:3] = self.pose_acc_ang_r
        xacc_ang[3:6] = self.pose_acc_ang_l
        return xacc_ang
    
    @property
    def effector_wrc(self):
        xwrc = np.zeros(12)
        xwrc[0:6] = self.pose_wrench_r
        xwrc[6:12] = self.pose_wrench_l
        return xwrc
    
    # TODO refactor this
    @effector_wrc.setter
    def effector_wrc(self, value):
        self._right.effector_wrc = value[:6]
        self._left.effector_wrc = value[6:]
    
    @property
    def effector_force(self):
        xforce = np.zeros(6)
        xforce[0:3] = self.pose_force_r
        xforce[3:6] = self.pose_force_l
        return xforce
    
    @property
    def effector_moment(self):
        xmoment = np.zeros(6)
        xmoment[0:3] = self.pose_torque_r
        xmoment[3:6] = self.pose_torque_l
        return xmoment
    
    @property
    def jacobian(self):
        return jacobian_combine(self.jacobian_r, self.jacobian_l)
    
    # TODO refactor this
    @jacobian.setter
    def jacobian(self, value):
        self._right.jacobian = value[:6, :7]
        self._left.jacobian = value[6:, 7:]


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
