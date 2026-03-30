from typing import Tuple, Optional

import numpy as np
import quaternion as quat


class RobotState(object):
    """ Class for storing the joint state
    """
    def __init__(
        self,
        dofs: int,
        # TODO this should become a topology... yeah sure
        jnt_pos: np.ndarray = None,
        jnt_vel: np.ndarray = None,
        jnt_acc: np.ndarray = None,
        jnt_tau: np.ndarray = None,
        # TODO this should become a Frame
        eff_pos: np.ndarray = np.zeros(3),
        eff_rot: np.quaternion = quat.one,
        eff_vel: np.ndarray = np.zeros(6),
        eff_acc: np.ndarray = np.zeros(6),
        eff_wrc: np.ndarray = np.zeros(6),
        jac: np.ndarray = None,
        jac_dt: np.ndarray = None
    ):
        """ Store state of a robot.
            All the variables are logically related as:
                - [p,Q] = K(q)       with K(.) being the direct kinematics function
                - v = J*dq 
                - a = dJ*dq + J*ddq
                - τ = JT*γ           with  JT  being J transposed
            ATTENTION: when reading or updating variables, beware that related 
                variables are not automatically updated, so all related variables 
                must be updated manually.

            :param dofs: degrees of freedom of the robot ( joint space size: n )
            :param jnt_pos: joint positions ( in configuration space: q ∈ R^n )
            :param jnt_vel: joint velocities ( in configuration space: dq ∈ R^n )
            :param jnt_acc: joint accelerations ( in configuration space: ddq ∈ R^n )
            :param jnt_tau: exogenous joint torques ( in configuration space: τ ∈ R^n )
            :param eff_pos: effector position ( in operational space: p ∈ R^3 )
            :param eff_rot: effector rotation ( in operational space: Q ∈ H )
            :param eff_vel: effector velocities ( in operational space: v=[dp, ω] ∈ R^3 x R^3 )
            :param eff_acc: effector accelerations ( in operational space: a=[ddp, dω] ∈ R^3 x R^3 )
            :param eff_wrc: exogenous effector wrench ( in operational space: γ=[f,μ] ∈ R^3 x R^3 )
            :param jac: base to effector Jacobian ( J ∈ R^6xn )
            :param jac_dt: base to effector Jacobian time-derivative ( dJ ∈ R^6xn )
        """
        self.dofs = dofs
        # joint space
        self.joint_pos = self._check_shape(jnt_pos, (self.dofs,))
        self.joint_vel = self._check_shape(jnt_vel, (self.dofs,))
        self.joint_acc = self._check_shape(jnt_acc, (self.dofs,))
        self.joint_tau = self._check_shape(jnt_tau, (self.dofs,))
        # cartesian space
        self.effector_pos = self._check_shape(eff_pos, (3,))
        self.effector_rot = eff_rot
        self.effector_vel = self._check_shape(eff_vel, (6,))
        self.effector_acc = self._check_shape(eff_acc, (6,))
        self.effector_wrc = self._check_shape(eff_wrc, (6,))
        # jacobians
        self.jacobian = self._check_shape(jac, (6, self.dofs))
        self.jacobian_dt = self._check_shape(jac_dt, (6, self.dofs))

    @staticmethod
    def _check_shape(array: Optional[np.ndarray], shape: Tuple):
        if array is not None:
            if array.shape != shape:
                raise ValueError(f"Shapes inconsistent with given space (expected {shape}, found {array.shape})")
            else:
                return array
        else:
            return np.zeros(shape, dtype=np.float64)

    @property
    def effector_vel_lin(self):
        return self.effector_vel[:3]

    @effector_vel_lin.setter
    def effector_vel_lin(self, effector_vel_lin: np.ndarray):
        self.effector_vel[:3] = effector_vel_lin

    @property
    def effector_vel_ang(self):
        return self.effector_vel[3:]

    @effector_vel_ang.setter
    def effector_vel_ang(self, effector_vel_ang: np.ndarray):
        self.effector_vel[3:] = effector_vel_ang

    @property
    def effector_acc_lin(self):
        return self.effector_acc[:3]

    @effector_acc_lin.setter
    def effector_acc_lin(self, effector_acc_lin: np.ndarray):
        self.effector_acc[:3] = effector_acc_lin

    @property
    def effector_acc_ang(self):
        return self.effector_acc[3:]

    @effector_acc_ang.setter
    def effector_acc_ang(self, effector_acc_ang: np.ndarray):
        self.effector_acc[3:] = effector_acc_ang

    @property
    def effector_force(self):
        return self.effector_wrc[:3]

    @effector_force.setter
    def effector_force(self, effector_force: np.ndarray):
        self.effector_wrc[:3] = effector_force

    @property
    def effector_moment(self):
        return self.effector_wrc[3:]

    @effector_moment.setter
    def effector_moment(self, effector_moment: np.ndarray):
        self.effector_wrc[3:] = effector_moment