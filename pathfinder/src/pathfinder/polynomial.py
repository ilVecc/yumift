from typing import Tuple

import numpy as np, quaternion as quat

from .base import TParam, Trajectory, MultiTrajectory
from .base_impl import PositionParam, PoseParam, QuaternionParam

from dynamicals.utils.quaternions import quat_is_closest


################################################################################
##                                 TRAJECTORY                                 ##
################################################################################

class CubicTrajectory(Trajectory[TParam]):
    def __init__(self) -> None:
        self._coeffs : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        super().__init__()
    
    def clear(self) -> None:
        self._coeffs = None
        super().clear()
    
    @staticmethod
    def calculate_coefficients(xi: np.ndarray, dxi: np.ndarray, xf: np.ndarray, dxf: np.ndarray, tf: float)  -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Calculate cubic coefficients
            :param xi: initial state
            :param dxi: initial velocity
            :param xf: final state
            :param dxf: final velocity
            :param tf: total time of the trajectory
        """
        a0 = xi
        a1 = dxi
        a2 = (3*xf - dxf*tf - 2*a1*tf - 3*a0)/(tf**2)
        a3 = (dxf - (2*a2*tf + a1))/(3*tf**2)
        return (a0, a1, a2, a3)
    
    @staticmethod
    def evaluate_at(coeffs : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Calculate position, velocity and acceleration at a time given the cubic coefficients
            :param a0: 0th order coefficients
            :param a1: 1st order coefficients
            :param a2: 2nd order coefficients
            :param a3: 3rd order coefficients
            :param t: current time 0 <= t <= tf
        """
        a0, a1, a2, a3 = coeffs
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
    def compute(xi: np.ndarray, dxi: np.ndarray, xf: np.ndarray, dxf: np.ndarray, tf: float, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Calculate position, velocity and acceleration given the instant
            :param xi: initial state
            :param dxi: initial velocity
            :param xf: final state
            :param dxf: final velocity
            :param tf: total time of the trajectory
            :param t: current time 0 <= t <= tf
        """
        coeffs = CubicPosTrajectory.calculate_coefficients(xi, dxi, xf, dxf, tf)
        x, dx, ddx = CubicPosTrajectory.evaluate_at(coeffs, t)
        return x, dx, ddx


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
        self._coeffs = CubicTrajectory.calculate_coefficients(pos_init.pos, pos_init.vel, pos_final.pos, pos_final.vel, tf)
    
    def compute(self, t) -> PositionParam:
        """ Calculate target position, velocity and acceleration using  compute_trajectory
            :param t: current time 0 <= t <= tf (float)
        """
        t = max(0, min(t, self._duration))  # np.clip is 3 times slower o.o
        x, dx, ddx = CubicTrajectory.evaluate_at(self._coeffs, t)
        return PositionParam(x, dx, ddx)

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
        if not quat_is_closest(quat_init.quat, quat_final.quat):
            # HACK at least it feels like one
            quat_final.quat = -quat_final.quat
            # quat_init.quat = -quat_init.quat
        # `quat.as_rotation_vector` is slow, do it manually
        # 2*log(q=[cos(α/2) sin(α/2)n]) = 2*[0 (α/2)n] --> r = αn 
        ri = 2*np.log(quat_init.quat).vec
        rf = 2*np.log(quat_final.quat).vec
        self._coeffs = CubicTrajectory.calculate_coefficients(ri, quat_init.vel, rf, quat_final.vel, tf)
    
    def compute(self, t: float) -> QuaternionParam:
        t = max(0, min(t, self._duration))  # np.clip is 3 times slower o.o
        r, w, dw = CubicTrajectory.evaluate_at(self._coeffs, t)
        # `quat.from_rotation_vector` is slow, do it manually
        q = np.exp(quat.quaternion(0, *(0.5*r)))
        return QuaternionParam(q, w, dw)

class CubicPoseTrajectory(CubicTrajectory[PoseParam]):
    def __init__(self) -> None:
        self._traj_pos = CubicPosTrajectory()
        self._traj_rot = CubicQuatTrajectory()
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
        
        self._out_param.pos = pos_param.pos
        self._out_param.rot = quat_param.quat
        self._out_param._fields[1][0:3] = pos_param._fields[1][0:3]
        self._out_param._fields[1][3:6] = quat_param._fields[1][0:3]
        self._out_param._fields[2][0:3] = pos_param._fields[2][0:3]
        self._out_param._fields[2][3:6] = quat_param._fields[2][0:3]
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
    