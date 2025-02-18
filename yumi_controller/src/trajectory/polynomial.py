from typing import Tuple

import numpy as np
import quaternion as quat

from .base import TParam, Trajectory, MultiTrajectory
from .base_impl import PositionParam, PoseParam, QuaternionParam

from dynamics.quat_utils import quat_diff
from dynamics.utils import norm3, normalize3


################################################################################
##                                 TRAJECTORY                                 ##
################################################################################

class CubicTrajectory(Trajectory[TParam]):
    def __init__(self) -> None:
        self._a0: np.ndarray
        self._a1: np.ndarray
        self._a2: np.ndarray
        self._a3: np.ndarray
        super().__init__()
    
    def clear(self) -> None:
        self._a0 = None
        self._a1 = None
        self._a2 = None
        self._a3 = None
        super().clear()
    
    @staticmethod
    def calculate_coefficients(xi: np.ndarray, dxi: np.ndarray, xf: np.ndarray, dxf: np.ndarray, tf: float)  -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Calculate cubic coefficients
            :param qi: initial state
            :param dqi: initial velocity
            :param qf: final state
            :param dqf: final velocity
            :param tf: total time of the trajectory
        """
        a0 = xi
        a1 = dxi
        a2 = (3*xf - dxf*tf - 2*a1*tf - 3*a0)/(tf**2)
        a3 = (dxf - (2*a2*tf + a1))/(3*tf**2)
        return a0, a1, a2, a3
    
    @staticmethod
    def calculate_trajectory(a0: np.ndarray, a1: np.ndarray, a2: np.ndarray, a3: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Calculate position, velocity and acceleration given the cubic coefficients
            :param a0: 0th order coefficient
            :param a1: 1st order coefficient
            :param a2: 2nd order coefficient
            :param a3: 3rd order coefficient
            :param t: current time 0 <= t <= tf
        """
        x = ((a3*t + a2)*t + a1)*t + a0
        dx = (3*a3*t + 2*a2)*t + a1
        ddx = (6*a3)*t + 2*a2
        # these down here should be optimal but actually are pure Python and 
        # use for-loop, which instead worsen performance a little
        # x = np.polynomial.polynomial.polyval(t, [a0, a1, a2, a3])
        # dx = np.polynomial.polynomial.polyval(t, [a1, 2*a2, 3*a3])
        # ddx = np.polynomial.polynomial.polyval(t, [2*a2, 6*a3])
        return x, dx, ddx
    
    @staticmethod
    def compute_trajectory(xi: np.ndarray, dxi: np.ndarray, xf: np.ndarray, dxf: np.ndarray, tf: float, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Calculate position, velocity and acceleration given the instant
            :param qi: initial state
            :param dqi: initial velocity
            :param qf: final state
            :param dqf: final velocity
            :param tf: total time of the trajectory
            :param t: current time 0 <= t <= tf
        """
        a0, a1, a2, a3 = CubicPosTrajectory.calculate_coefficients(xi, dxi, xf, dxf, tf)
        q, dq, ddq = CubicPosTrajectory.calculate_trajectory(a0, a1, a2, a3, t)
        return q, dq, ddq


class CubicPosTrajectory(CubicTrajectory[PositionParam]):
    
    def __init__(self) -> None:
        super().__init__()
    
    def update(self, pos_init: PositionParam, pos_final: PositionParam, tf: float) -> None:
        """ Set up internal coefficients using  calculate_coefficients
            :param pos_init: initial position parameter
            :param pos_final: final position parameter
            :param tf: total time of the trajectory
        """
        super().update(pos_init, pos_final, tf)
        self._a0, self._a1, self._a2, self._a3 = CubicTrajectory.calculate_coefficients(pos_init.pos, pos_init.vel, pos_final.pos, pos_final.vel, tf)
    
    def compute(self, t) -> PositionParam:
        """ Calculate target position, velocity and acceleration using  compute_trajectory
            :param t: current time 0 <= t <= tf (float)
        """
        t = max(0, min(t, self._duration))  # np.clip is 3 times slower o.o
        x, dx, ddx = CubicTrajectory.calculate_trajectory(self._a0, self._a1, self._a2, self._a3, t)
        return PositionParam(x, dx, ddx)





class CubicQuatTrajectory_OLD(CubicTrajectory[QuaternionParam]):
    
    def __init__(self) -> None:
        self._r: np.ndarray
        self._qi: np.quaternion
        super().__init__()
    
    def clear(self) -> None:
        self._r = None
        self._qi = None
        super().clear()
    
    def update(self, quat_init: QuaternionParam, quat_final: QuaternionParam, tf: float) -> None:
        """ Set up internal coefficients using  calculate_coefficients  
            :param quat_init: initial quaternion parameter
            :param quat_init: final quaternion parameter
            :param tf: total time of the trajectory
        """
        super().update(quat_init, quat_final, tf)
        qi: np.quaternions = quat_init.quat
        qf: np.quaternions = quat_final.quat
        # use minimum distance
        self._qi = qi
        qr = quat_diff(self._qi, qf)
        self._r, vf = normalize3(2*np.log(qr).vec, return_norm=True)  # `quat.as_rotation_vector()` === `2*np.log().vec`
        # TODO add velocity handling (use quaternion trajectory planning)
        self._a0, self._a1, self._a2, self._a3 = CubicTrajectory.calculate_coefficients(0, 0, vf, 0, tf)
    
    def compute(self, t: float) -> QuaternionParam:
        t = max(0, min(t, self._duration))  # np.clip is 3 times slower o.o
        v, dv, ddv = CubicTrajectory.calculate_trajectory(self._a0, self._a1, self._a2, self._a3, t)
        
        qr = np.exp(quat.quaternion(0, *(v * self._r)) / 2)  # `quat.from_rotation_vector` is slow, do it manually
        qe = qr * self._qi
        w = quat.quaternion(0, *(dv * self._r))  # `quat.from_vector_part` is ultra slow, do it manually
        we = (self._qi * w * self._qi.conj()).vec
        dw = quat.quaternion(0, *(ddv * self._r))
        dwe = (self._qi * dw * self._qi.conj()).vec
        
        return QuaternionParam(qe, we, dwe)


import dynamics.quat_utils as quat_utils
class CubicQuatTrajectory(CubicTrajectory[QuaternionParam]):
    
    def __init__(self) -> None:
        super().__init__()
    
    def update(self, quat_init: QuaternionParam, quat_final: QuaternionParam, tf: float) -> None:
        """ Set up internal coefficients using  calculate_coefficients  
            :param quat_init: initial quaternion parameter
            :param quat_init: final quaternion parameter
            :param tf: total time of the trajectory
        """
        super().update(quat_init, quat_final, tf)
        self._a0, self._a1, self._a2, self._a3 = CubicTrajectory.calculate_coefficients(quat_init.quat.log().vec, quat_init.vel, quat_final.quat.log().vec, quat_final.vel, tf)
    
    def compute(self, t: float) -> QuaternionParam:
        t = max(0, min(t, self._duration))  # np.clip is 3 times slower o.o
        q, dq, ddq = CubicTrajectory.calculate_trajectory(self._a0, self._a1, self._a2, self._a3, t)
        
        Q = np.exp(quat.quaternion(0, *q) / 2)  # `quat.from_rotation_vector` is slow, do it manually
        Jq = quat_utils.jac_q(q)
        Jq_dq = quat.quaternion(*(Jq @ dq))
        W = 2 * Jq_dq * Q.conj()
        Jq_ddq = quat.quaternion(*(Jq @ ddq))
        dW = W + 2 * Jq_ddq * Q.conj() - 0.5 * norm3(W.vec)**2 * quat.one

        return QuaternionParam(Q, W.vec, dW.vec)





class CubicPoseTrajectory(CubicTrajectory[PoseParam]):
    def __init__(self) -> None:
        self._traj_pos = CubicPosTrajectory()
        self._traj_rot = CubicQuatTrajectory_OLD()
        # creating the param every time requires concatenation of velocities and
        # accelerations, which is very expensive, so simply fill the values here
        self._out_param = PoseParam(np.zeros(3), quat.one, np.zeros(6))
        super().__init__()
    
    def clear(self) -> None:
        self._traj_pos.clear()
        self._traj_rot.clear()
        super().clear()

    def update(self, pose_init: PoseParam, pose_final: PoseParam, tf: float) -> None:
        super().update(pose_init, pose_final, tf)
        self._traj_pos.update(pose_init.as_pos_param(), pose_final.as_pos_param(), tf)
        self._traj_rot.update(pose_init.as_quat_param(), pose_final.as_quat_param(), tf)
    
    def compute(self, t) -> PoseParam:
        pos_param = self._traj_pos.compute(t)  # ca 10kHz
        quat_param = self._traj_rot.compute(t)  # ca 900Hz
        
        # TODO slicing arrays is apparently slow...
        self._out_param.pos = pos_param.pos
        self._out_param.rot = quat_param.quat
        self._out_param._fields[1][0] = pos_param._fields[1][0]
        self._out_param._fields[1][1] = pos_param._fields[1][1]
        self._out_param._fields[1][2] = pos_param._fields[1][2]
        self._out_param._fields[1][3] = quat_param._fields[1][0]
        self._out_param._fields[1][4] = quat_param._fields[1][1]
        self._out_param._fields[1][5] = quat_param._fields[1][2]
        self._out_param._fields[2][0] = pos_param._fields[2][0]
        self._out_param._fields[2][1] = pos_param._fields[2][1]
        self._out_param._fields[2][2] = pos_param._fields[2][2]
        self._out_param._fields[2][3] = quat_param._fields[2][0]
        self._out_param._fields[2][4] = quat_param._fields[2][1]
        self._out_param._fields[2][5] = quat_param._fields[2][2]
        return self._out_param


################################################################################
##                                    PATH                                    ##
################################################################################

class CubicPath(MultiTrajectory[TParam]):
    def __init__(self, trajectory: CubicTrajectory[TParam]) -> None:
        super().__init__(trajectory)

class CubicPosePath(CubicPath[PoseParam]):
    """ Shorthand for `MultiTrajectory[PoseParam](CubicPoseTrajectory())` """
    def __init__(self, trajectory: CubicTrajectory[PoseParam] = None) -> None:
        if trajectory is None:
            super().__init__(CubicPoseTrajectory())
        else:
            super().__init__(trajectory)
    