import numpy as np
import quaternion as quat


def skew_matrix(vector) -> np.ndarray:
    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])

def norm3(v: np.ndarray):
    """ Fast 3-vectory norm. Twice as fast as `np.linalg.norm()`
        :param v: the vector for the norm operation
    """
    return np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

def norm4(v: np.ndarray):
    """ Fast 4-vectory norm. Twice as fast as `np.linalg.norm()`
        :param v: the vector for the norm operation
    """
    return np.sqrt(v[0]**2 + v[1]**2 + v[2]**2 + v[3]**2)

def normalize3(v: np.ndarray, return_norm=False) -> np.ndarray:
    """ Calculates the normalized vector
        :param v: the vector to normalize
        :param return_norm: whether to return the vector norm or not 
    """
    norm = norm3(v)
    w = v / (norm or 1)
    if return_norm:
        return w, norm
    return w


from .quaternions import quat_avg
from .jacobians import jacobian_change_end_frame, skew_matrix

class Frame(object):
    """ Reference frame or transformation
    """
    def __init__(
        self,
        position: np.ndarray = np.zeros(3),
        rotation: np.quaternion = quat.one,
        velocity: np.ndarray = np.zeros(6),
        acceleration: np.ndarray = np.zeros(6),
        wrench: np.ndarray = np.zeros(6)
    ):
        """ Initialize a frame with position, rotation, and velocity
            :param position: np.array([x,y,z]) position [m]
            :param rotation: np.quaternion([w,x,y,z]) orientation [unit quaternion]
        """
        self._pos = position
        self._quat = rotation
        self._vel = velocity
        # TODO i don't know, implement me...
        self._acc = acceleration
        self._wrc = wrench

    def __add__(self, other: "Frame") -> "Frame":
        """ Component-wise addition of this frame to the other.

            ATTENTION: order matters (rotations are non-commutative).
        """
        return Frame(
            self._pos + other._pos,
            self._quat * other._quat,
            self._vel + other._vel)

    def __sub__(self, other: "Frame") -> "Frame":
        """ Component-wise subtraction from this frame by the other.

            ATTENTION: order matters (rotations are non-commutative).
        """
        return Frame(
            self._pos - other._pos,
            self._quat * other._quat.conjugate(),
            self._vel - other._vel)

    def __matmul__(self, frame: "Frame") -> "Frame":
        """ Perform this transformation on the other transformation/frame. 
            This operation makes sense when `self` is a transformation wrt `frame`.
            Being a transformation, `self.vel` should be `0`, but can actually 
            be anything to be added to `frame`'s transformed velocity.
        """
        return Frame(
            # t := q1 * t2 * ~q1 + t1
            position=(self.rot * quat.quaternion(0, *(frame.pos)) * self.rot.conjugate()).vec + self.pos,
            # q := q1 * q2
            rotation=self.rot * frame.rot,
            # v := v1 + J*v2
            velocity=self.vel + jacobian_change_end_frame(self.pos) @ frame.vel)

    def inv(self) -> "Frame":
        """ Invert this transformation
        """
        return Frame(
            # t := - (~q) * t * ~(~q)
            position=-(self.rot.conjugate() * quat.quaternion(0, *(self.pos)) * self.rot).vec,
            # q := ~q
            rotation=self.rot.conjugate(),
            # v := -v
            velocity=-self.vel)

    def __invert__(self) -> "Frame":
        return self.inv()

    def __truediv__(self, other: "Frame") -> "Frame":
        return self @ other.inv()

    @staticmethod
    def avg(*poses : "Frame"):
        """ Average multiple poses.
            This operation makes sense only when the provided poses are expressed in the same base frame.
        """
        return Frame(
            position=np.mean(np.stack([p._pos for p in poses]), axis=0),
            rotation=quat_avg(np.stack([p._quat for p in poses])),
            velocity=np.mean(np.stack([p._vel for p in poses]), axis=0))

    @property
    def pos(self) -> np.ndarray:
        """ Returns the position
        """
        return self._pos

    @pos.setter
    def pos(self, position: np.ndarray):
        """ Updates the position
            :param position: np.array([x,y,z]) [m]
        """
        self._pos = position

    @property
    def rot(self) -> np.quaternion:
        """ Returns the quaternion
        """
        return self._quat

    @rot.setter
    def rot(self, quaternion: np.quaternion):
        """ Updates the orientation
            :param quaternion: np.quaternion([w,x,y,z]) orientation [unit quaternion]
        """
        self._quat = quaternion

    @property
    def vel(self):
        return self._vel

    @vel.setter
    def vel(self, velocity: np.ndarray):
        assert velocity.shape == (6,)
        self._vel = velocity

    @property
    def acc(self):
        return self._acc

    @acc.setter
    def acc(self, acceleration: np.ndarray):
        assert acceleration.shape == (6,)
        self._acc = acceleration

    @property
    def wrc(self):
        return self._wrc

    @wrc.setter
    def wrc(self, wrench: np.ndarray):
        assert wrench.shape == (6,)
        self._wrc = wrench

    def adjoint(self):
        rot = quat.as_rotation_matrix(self._quat)
        adj = np.zeros((6,6))
        adj[0:3,0:3] = rot
        adj[3:6,3:6] = rot
        adj[0:3,3:6] = skew_matrix(self._pos) @ rot
        return adj

    def partial(self, alpha: float):
        # `quat.as_rotation_vector()` === `2*np.log().vec` and since `quat_diff` is 
        # normalized, the `.vec` is not necessary because the scalar part will be 0. 
        # then, since `quat.from_rotation_vector(...)` === `np.exp([0, .../2])`, we 
        # can simplify the 2s and avoid pre-pending the 0
        return Frame(position=alpha * self._pos,
                     rotation=np.exp(alpha * np.log(self._quat)))

    @staticmethod
    def from_matrix(matrix: np.ndarray) -> "Frame":
        """ Create a Frame object from a homogenerous matrix
            :param matrix: the homogeneous matrix
        """
        assert matrix.shape == (4, 4) and np.all(matrix[3,:] == np.array([0, 0, 0, 1])) , "Matrix must be homogeneous"
        return Frame(matrix[:3, 3], quat.from_rotation_matrix(matrix[:3,:3]))

    def matrix(self):
        """ Convert this Frame object to a homogeneous matrix representation
        """
        mat = np.eye(4)
        mat[:3,:3] = quat.as_rotation_matrix(self._quat)
        mat[:3, 3] = self._pos
        return mat

    def __repr__(self) -> str:
        return f"pos: {np.array_str(self.pos, precision=2, suppress_small=True)}" \
             + f"rot: {np.array_str(quat.as_float_array(self.rot), precision=2, suppress_small=True)}" \
             + f"vel: {np.array_str(self.vel, precision=2, suppress_small=True)}"
