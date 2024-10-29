from typing import Tuple

import numpy as np
import quaternion as quat

from .base import TParam, Trajectory, Path
from .base_impl import PositionParam, PoseParam, QuaternionParam

from dynamics.quat_utils import quat_min_diff


def normalize(v: np.ndarray, return_norm=False):
    """ Calculates the normalized vector
        :param v: the array to normalize
    """
    norm = np.linalg.norm(v)
    w = v / (norm + (norm == 0))
    if return_norm:
        return w, norm 
    return w

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
        t = np.clip(t, 0, self._duration)
        x, dx, ddx = CubicTrajectory.calculate_trajectory(self._a0, self._a1, self._a2, self._a3, t)
        return PositionParam(x, dx, ddx)

class CubicQuatTrajectory(CubicTrajectory[QuaternionParam]):
    
    def __init__(self) -> None:
        self._r: np.ndarray
        self._qi: np.quaternion
        super().__init__()
    
    def clear(self) -> None:
        self._r = None
        self._qi = None
        super().clear()
    
    # TODO add velocity handling (use quaternion trajectory planning)
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
        qr = quat_min_diff(self._qi, qf)
        self._r, vf = normalize(quat.as_rotation_vector(qr), return_norm=True)
        self._a0, self._a1, self._a2, self._a3 = CubicTrajectory.calculate_coefficients(0, 0, vf, 0, tf)
    
    def compute(self, t: float) -> QuaternionParam:
        t = np.clip(t, 0, self._duration)
        v, dv, ddv = CubicTrajectory.calculate_trajectory(self._a0, self._a1, self._a2, self._a3, t)
        qr = quat.from_rotation_vector(v * self._r)
        qe = qr * self._qi
        w = quat.from_vector_part(dv * self._r)
        we = quat.as_vector_part(self._qi * w * self._qi.conjugate())
        dw = quat.from_vector_part(ddv * self._r)
        dwe = quat.as_vector_part(self._qi * dw * self._qi.conjugate())
        return QuaternionParam(qe, we, dwe)

class CubicPoseTrajectory(CubicTrajectory[PoseParam]):
    def __init__(self) -> None:
        self._traj_pos = CubicPosTrajectory()
        self._traj_rot = CubicQuatTrajectory()
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
        pos_param = self._traj_pos.compute(t)
        quat_param = self._traj_rot.compute(t)
        vel = np.concatenate([pos_param.vel, quat_param.vel])
        acc = np.concatenate([pos_param.acc, quat_param.acc])
        return PoseParam(pos_param.pos, quat_param.quat, vel, acc)


################################################################################
##                                    PATH                                    ##
################################################################################

class CubicPath(Path[TParam]):
    def __init__(self, trajectory: CubicTrajectory[TParam]) -> None:
        super().__init__(trajectory)

class CubicPosePath(CubicPath[PoseParam]):
    """ Shorthand for  Path[PoseParam](CubicPoseTrajectory()) """
    def __init__(self, trajectory: CubicTrajectory[PoseParam] = None) -> None:
        if trajectory is None:
            super().__init__(CubicPoseTrajectory())
        else:
            super().__init__(trajectory)


if __name__ == "__main__":
    from .plotter import plot_quat
    
    vi = np.pi/2 * np.array([0, 0.707, 0.707])
    vf = np.pi/2 * np.array([0.707, 0.707, 0])
    
    traj = CubicQuatTrajectory()
    qi = QuaternionParam(quat.from_rotation_vector(vi), np.zeros(3))
    qf = QuaternionParam(quat.from_rotation_vector(vf), np.zeros(3))
    traj.update(qi, qf, tf=4)

    out = []
    for t in np.linspace(0, 4, 100, endpoint=True):
        param = traj.compute(t)
        out.append(param.quat)
    
    out = np.array(out)
    plot_quat(out, "trajectory")
    
    