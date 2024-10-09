import numpy as np
import quaternion as quat

from .base import Param


class JointParam(Param):
    def __init__(self, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray) -> None:
        assert q.shape == dq.shape == ddq.shape == (q.size,), "All parameters must be vectors of same shape"
        super().__init__(q, dq, ddq)

    @property
    def q(self) -> np.ndarray:
        return self.value

    @property
    def dq(self) -> np.ndarray:
        return self.speed

    @property
    def ddq(self) -> np.ndarray:
        return self.curve

class PositionParam(Param):
    def __init__(self, position: np.ndarray, velocity: np.ndarray, acceleration: np.ndarray = np.zeros(3)) -> None:
        assert position.shape == (3,)
        assert velocity.shape == (3,)
        assert acceleration.shape == (3,)
        super().__init__(position, velocity, acceleration)

    @property
    def pos(self) -> np.ndarray:
        return self.value

    @property
    def vel(self) -> np.ndarray:
        return self.speed

    @property
    def acc(self) -> np.ndarray:
        return self.curve

class QuaternionParam(Param):
    def __init__(self, quaternion: np.quaternion, angular_velocity: np.ndarray, angular_acceleration: np.ndarray = np.zeros(3)) -> None:
        assert angular_velocity.shape == (3,)
        assert angular_acceleration.shape == (3,)
        super().__init__(quaternion, angular_velocity, angular_acceleration)

    @property
    def quat(self) -> np.quaternion:
        return self.value

    @property
    def vel(self) -> np.ndarray:
        return self.speed
    
    @property
    def acc(self) -> np.ndarray:
        return self.curve

class PoseParam(Param):
    def __init__(self, position: np.ndarray, rotation: np.quaternion, velocity: np.ndarray, acceleration: np.ndarray = np.zeros(6)) -> None:
        assert position.shape == (3,)
        assert velocity is None or velocity.shape == (6,)
        assert acceleration is None or acceleration.shape == (6,)
        super().__init__([position, rotation], velocity, acceleration)

    @property
    def pos(self) -> np.ndarray:
        return self.value[0]

    @property
    def rot(self) -> np.quaternion:
        return self.value[1]

    @property
    def vel(self) -> np.ndarray:
        return self.speed

    @vel.setter
    def vel(self, velocity) -> None:
        self._fields[1] = velocity

    @property
    def vel_lin(self) -> np.ndarray:
        return self.vel[0:3]

    @property
    def vel_ang(self) -> np.ndarray:
        return self.vel[3:6]

    @property
    def acc_lin(self) -> np.ndarray:
        return self.curve[0:3]

    @property
    def acc_ang(self) -> np.ndarray:
        return self.curve[3:6]

    def as_pos_param(self) -> PositionParam:
        return PositionParam(self.pos, self.vel_lin, self.acc_lin)

    def as_quat_param(self) -> QuaternionParam:
        return QuaternionParam(self.rot, self.vel_ang, self.acc_ang)
