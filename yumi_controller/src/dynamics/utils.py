from typing import Tuple

import numpy as np
import quaternion as quat

from dynamics.quat_utils import quat_avg


class Frame(object):
    """
    Reference frame or transformation
    """
    def __init__(self, position: np.ndarray = np.zeros(3), quaternion: np.quaternion = quat.one, velocity: np.ndarray = np.zeros(6)):
        """ Initialize a frame with position, rotation, and velocity
            :param position: np.array([x,y,z]) position [m]
            :param quaternion: np.quaternion([w,x,y,z]) orientation [unit quaternion]
        """
        self._pos = position
        self._quat = quaternion
        self._vel = velocity
    
    def __add__(self, other: "Frame") -> "Frame":
        """
        Component-wise addition of this frame to the other.
        ATTENTION: order matters (rotations are non-commutative).
        """
        return Frame(
            self._pos + other._pos, 
            self._quat * other._quat,
            self._vel + other._vel)

    def __sub__(self, other: "Frame") -> "Frame":
        """
        Component-wise subtraction from this frame by the other.
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
            position=quat.as_vector_part(self.rot * quat.from_vector_part(frame.pos) * self.rot.conjugate()) + self.pos,
            # q := q1 * q2
            quaternion=self.rot * frame.rot,
            # v := v1 + J*v2
            velocity=self.vel + jacobian_change_end_frame(self.pos) @ frame.vel)
    
    def inv(self) -> "Frame":
        """ Invert this transformation
        """
        return Frame(
            # t := - (~q) * t * ~(~q)
            position=-quat.as_vector_part(self.rot.conjugate() * quat.from_vector_part(self.pos) * self.rot),
            # q := ~q
            quaternion=self.rot.conjugate(),
            # v := -v
            velocity=-self.vel)  # TODO is this correct?

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
            quaternion=quat_avg(np.stack([p._quat for p in poses])),
            velocity=np.mean(np.stack([p._vel for p in poses]), axis=0))
        
    @property
    def pos(self):
        """ Returns the position
        """
        return self._pos

    @pos.setter
    def pos(self, position):
        """ Updates the position
            :param position: np.array([x,y,z]) [m]
        """
        self._pos = position

    @property
    def rot(self):
        """ Returns the quaternion
        """
        return self._quat
    
    @rot.setter
    def rot(self, quaternion):
        """ Updates the orientation
            :param quaternion: np.quaternion([w,x,y,z]) orientation [unit quaternion]
        """
        self._quat = quaternion
    
    @property
    def vel(self):
        return self._vel
    
    @vel.setter
    def vel(self, velocity):
        self._vel = velocity
    
    @staticmethod
    def from_matrix(matrix) -> "Frame":
        return Frame(matrix[:3, 3], quat.from_rotation_matrix(matrix[:3,:3]))
    
    def to_matrix(self):
        mat = np.eye(4)
        mat[:3,:3] = quat.as_rotation_matrix(self._quat)
        mat[:3, 3] = self._pos
        return mat
    
    def __repr__(self) -> str:
        return f"pos: {np.array_str(self.pos, precision=2, suppress_small=True)}  " \
             + f"rot: {np.array_str(quat.as_float_array(self.rot), precision=2, suppress_small=True)}"


class RobotState(object):
    """
    Class for storing the joint state
    """
    def __init__(
        self, 
        dofs: int,
        joint_pos: np.ndarray = None, 
        joint_vel: np.ndarray = None, 
        joint_acc: np.ndarray = None,
        joint_torque: np.ndarray = None,
        pose_pos: np.ndarray = np.zeros(3),
        pose_rot: np.quaternion = quat.one,
        pose_vel: np.ndarray = np.zeros(6),
        pose_acc: np.ndarray = np.zeros(6),
        pose_wrench: np.ndarray = np.zeros(6),
        jacobian: np.ndarray = None,
        jacobian_derivative: np.ndarray = None
    ):
        """
        Store state of a robot.
        All the variables are logically related as:
            - [p,Q] = K(q)       with K(.) being the direct kinematics function
            - v = J*dq 
            - a = dJ*dq + J*ddq
            - τ = JT*γ           with  JT  being J transposed
        ATTENTION: when updating variables, beware that related variables are not updated,
                   so all related variables must be updated manually.
        :param dofs: degrees of freedom of the robot ( joint space size: n )
        :param joint_pos: joint positions ( in configuration space: q ∈ R^n )
        :param joint_vel: joint velocities ( in configuration space: dq ∈ R^n )
        :param joint_acc: joint accelerations ( in configuration space: ddq ∈ R^n )
        :param joint_torque: exogenous joint torques ( in configuration space: τ ∈ R^n )
        :param pose_pos: effector position ( in operational space: p ∈ R^3 )
        :param pose_rot: effector rotation ( in operational space: Q ∈ H )
        :param pose_vel: effector velocities ( in operational space: v=[dp, ω] ∈ R^3 x R^3 )
        :param pose_acc: effector accelerations ( in operational space: a=[ddp, dω] ∈ R^3 x R^3 )
        :param pose_wrench: exogenous effector force and moment ( in operational space: γ=[f,μ] ∈ R^3 x R^3 )
        :param jacobian: base to effector Jacobian ( J ∈ R^6xn )
        :param jacobian_derivative: base to effector Jacobian time-derivative ( dJ ∈ R^6xn )
        """
        self._dofs = dofs
        # joint space
        self._joint_pos = self._check_shape(joint_pos, (self._dofs,))
        self._joint_vel = self._check_shape(joint_vel, (self._dofs,))
        self._joint_acc = self._check_shape(joint_acc, (self._dofs,))
        self._joint_torque = self._check_shape(joint_torque, (self._dofs,))
        # cartesian space
        self._pose_pos = self._check_shape(pose_pos, (3,))
        self._pose_rot = pose_rot
        self._pose_vel = self._check_shape(pose_vel, (6,))
        self._pose_acc = self._check_shape(pose_acc, (6,))
        self._pose_wrench = self._check_shape(pose_wrench, (6,))
        # jacobians
        self._jacobian = self._check_shape(jacobian, (6, self._dofs))
        self._jacobian_derivative = self._check_shape(jacobian_derivative, (6, self._dofs))
    
    @staticmethod
    def _check_shape(array: np.ndarray, shape: Tuple):
        if array is not None:
            if array.shape != shape:
                raise ValueError("Shapes of inputs must be consistent with given dofs or cartesian space")
            else:
                return array
        else:
            return np.zeros(shape, dtype=np.float64)
    
    @property
    def dofs(self):
        return self._dofs
    
    @property
    def joint_pos(self):
        return self._joint_pos
    
    @property
    def joint_vel(self):
        return self._joint_vel
    
    @property
    def joint_acc(self):
        return self._joint_acc
    
    @property
    def joint_torque(self):
        return self._joint_torque

    @property
    def pose_pos(self):
        return self._pose_pos

    @property
    def pose_rot(self):
        return self._pose_rot
    
    @property
    def pose_vel(self):
        return self._pose_vel
    
    @property
    def pose_vel_lin(self):
        """ Returns the pose linear velocity """
        return self.pose_vel[:3]

    @property
    def pose_vel_ang(self):
        """ Returns the pose angular velocity """
        return self.pose_vel[3:]
    
    @property
    def pose_acc(self):
        return self._pose_acc
    
    @property
    def pose_acc_lin(self):
        """ Returns the pose linear acceleration """
        return self._pose_acc[:3]

    @property
    def pose_acc_ang(self):
        """ Returns the pose angular acceleration """
        return self._pose_acc[3:]
    
    @property
    def pose_wrench(self):
        return self._pose_wrench
    
    @property
    def pose_force(self):
        return self._pose_wrench[:3]
    
    @property
    def pose_torque(self):
        return self._pose_wrench[3:]
    
    @property
    def jacobian(self):
        return self._jacobian
    
    @property
    def jacobian_derivative(self):
        return self._jacobian_derivative


def jacobian_change_end_frame(dist_vec: np.ndarray, jacobian: np.ndarray = None) -> np.ndarray:
    """
    Extends the Jacobian with a new frame (changes end-effector)
    :param jacobian: the initial jacobian matrix
    :param dist_vec: relative vector from initial frame to desired frame wrt initial frame
    """
    #
    # J_BD = [[ I  -S(d_B_ED) ]  * J_BE
    #         [ 0          I  ]]
    # where
    #   J_BE    jacobian from  base B frame     to  effector E frame  expressed in  base B frame
    #   J_BD    jacobian from  base B frame     to  desired D frame   expressed in  base B frame
    #   d_B_DE  vector   from  effector E frame to  desired D frame   expressed in  base B frame
    #   S(.)    cross-product matrix
    #
    
    i3, z3, skew = np.eye(3), np.zeros((3,3)), skew_matrix(dist_vec)
    link_mat = np.block([[ i3, -skew ], 
                         [ z3,    i3 ]])
    
    if jacobian is not None:
        return link_mat @ jacobian
    else:
        return link_mat

def jacobian_change_base_frame(rot_quat: np.quaternion, jacobian: np.ndarray = None) -> np.ndarray:
    """
    Expresses the Jacobian from a new frame (changes base)
    :param jacobian: the initial jacobian matrix
    :param rot_quat: rotation quaternion from initial frame to desired frame wrt initial frame
    """
    #
    # J_FE = [[ R_FB     0 ]  * J_BE
    #         [    0  R_FB ]]
    # where
    #   J_BE    jacobian from  base B frame     to  effector E frame  expressed in  base B frame
    #   J_FE    jacobian from  generic F frame  to  effector E frame  expressed in  generic F frame  
    #   R_FB    rotation from  Base frame       to  generic F frame   expressed in  base B frame
    #
    
    z3, rot = np.zeros((3,3)), quat.as_rotation_matrix(rot_quat)
    link_mat = np.block([[ rot,  z3 ], 
                         [  z3, rot ]])
    
    if jacobian is not None:
        return link_mat @ jacobian
    else:
        return link_mat

def jacobian_change_frames(ee_dist_vec: np.ndarray, base_rot_quat: np.quaternion, jacobian: np.ndarray = None) -> np.ndarray:
    """
    Change simultaneously effector frame and base frame.
    :param ee_dist_vec: relative vector from initial effector frame to desired effector frame wrt initial effector frame
    :param base_rot_quat: rotation quaternion from initial base frame to desired base frame wrt initial base frame
    :param jacobian: the initial jacobian matrix
    """
    link_mat = jacobian_change_base_frame(base_rot_quat) @ jacobian_change_end_frame(ee_dist_vec)
    if jacobian is not None:
        return link_mat @ jacobian
    else:
        return link_mat

def jacobian_combine(*jacobians: np.ndarray) -> np.ndarray:
    """
    Combine jacobians in a block-diagonal matrix.
    """
    shapes = [(0, 0)] + [jac.shape for jac in jacobians]
    blocks = np.cumsum(shapes, axis=0)
    out = np.zeros(shape=blocks[-1,:])
    for idx in range(len(blocks)-1):
        x, y = shapes[idx]
        i, j = shapes[idx+1]
        out[x:x+i, y:y+j] = jacobians[idx]
    return out


def skew_matrix(vector) -> np.ndarray:
    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])

def normalize(v, return_norm=False) -> np.ndarray:
    """ Calculates the normalized vector
        :param v: the vector to normalize
        :param return_norm: whether to return the vector norm or not 
    """
    norm = np.linalg.norm(v)
    w = (v / norm) if norm != 0 else v
    if return_norm:
        return w, norm
    return w
