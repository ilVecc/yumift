from typing import Tuple, Optional

import numpy as np
import quaternion as quat

from .quat_utils import quat_avg


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
    def pos(self):
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
    def rot(self):
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
    
    def adjoint(self):
        rot = quat.as_rotation_matrix(self._quat)
        adj = np.zeros((6,6))
        adj[0:3,0:3] = rot
        adj[3:6,3:6] = rot
        adj[0:3,3:6] = skew_matrix(self._pos) @ rot
        return adj
    
    @staticmethod
    def from_matrix(matrix: np.ndarray) -> "Frame":
        """ Create a Frame object from a homogenerous matrix
            :param matrix: the homogeneous matrix
        """
        assert matrix.shape == (4, 4) and matrix[3,:] == np.array([0, 0, 0, 1]) , "Matrix must be homogeneous"
        return Frame(matrix[:3, 3], quat.from_rotation_matrix(matrix[:3,:3]))
    
    def to_matrix(self):
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


# TODO is this the wrong place for me?
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
                raise ValueError("Shapes of inputs must be consistent with given dofs or cartesian space")
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


def jacobian_change_end_frame(dist_vec: np.ndarray, jacobian: Optional[np.ndarray] = None) -> np.ndarray:
    """ Extends the Jacobian with a new frame (changes end-effector)
        :param dist_vec: relative vector from initial frame to desired frame wrt initial frame
        :param jacobian: the initial jacobian matrix
    """
    #
    # J_BD = [[ I  -S(d_B_ED) ]  * J_BE
    #         [ 0          I  ]]
    # where
    #   J_BE    jacobian from  base B frame     to  effector E frame  expressed in  base B frame
    #   J_BD    jacobian from  base B frame     to  desired D frame   expressed in  base B frame
    #   d_B_ED  vector   from  effector E frame to  desired D frame   expressed in  base B frame
    #   S(.)    cross-product matrix
    #
    
    link_mat = np.eye(6)
    link_mat[0:3,3:6] = -skew_matrix(dist_vec)
    
    if jacobian is not None:
        return link_mat @ jacobian
    else:
        return link_mat

def jacobian_change_base_frame(rot_quat: np.quaternion, jacobian: Optional[np.ndarray] = None) -> np.ndarray:
    """ Expresses the Jacobian from a new frame (changes base)
        :param rot_quat: rotation quaternion from desired frame to initial frame
        :param jacobian: the initial jacobian matrix
    """
    #
    # J_FE = [[ R_FB     0 ]  * J_BE
    #         [    0  R_FB ]]
    # where
    #   J_BE    jacobian from  base B frame     to  effector E frame  expressed in  base B frame
    #   J_FE    jacobian from  generic F frame  to  effector E frame  expressed in  generic F frame  
    #   R_FB    rotation from  generic F frame  to  Base frame
    #
    
    rot = quat.as_rotation_matrix(rot_quat)
    link_mat = np.zeros((6,6))
    link_mat[0:3,0:3] = rot
    link_mat[3:6,3:6] = rot
    
    if jacobian is not None:
        return link_mat @ jacobian
    else:
        return link_mat

def jacobian_change_frames(ee_dist_vec: np.ndarray, base_rot_quat: np.quaternion, jacobian: Optional[np.ndarray] = None) -> np.ndarray:
    """ Change simultaneously effector frame and base frame.
        :param ee_dist_vec: relative vector from initial effector frame to desired effector frame wrt initial base frame
        :param base_rot_quat: rotation quaternion from desired base frame to initial base frame
        :param jacobian: the initial jacobian matrix
    """
    link_mat = jacobian_change_base_frame(base_rot_quat) @ jacobian_change_end_frame(ee_dist_vec)
    if jacobian is not None:
        return link_mat @ jacobian
    else:
        return link_mat

def jacobian_combine(*jacobians: np.ndarray) -> np.ndarray:
    """ Combine jacobians in a block-diagonal matrix.
    """
    # optimize for the 2-jacobians case
    if len(jacobians) == 2:
        r1, c1 = jacobians[0].shape[0:2]
        r2, c2 = jacobians[1].shape[0:2]
        jac = np.zeros((r1 + r2, c1 + c2))
        jac[:r1, :c1] = jacobians[0]
        jac[r1:, c1:] = jacobians[1]
        return jac
    # optimize for the 3-jacobians case
    if len(jacobians) == 3:
        r1, c1 = jacobians[0].shape[0:2]
        r2, c2 = jacobians[1].shape[0:2]
        r3, c3 = jacobians[2].shape[0:2]
        jac = np.zeros((r1+r2+r3, c1+c2+c3))
        jac[:r1, :c1] = jacobians[0]
        jac[r1:r1+r2, c1:c1+c2] = jacobians[1]
        jac[r1+r2:, c1+c2:] = jacobians[2]
        return jac
    # deal with the general n-jacobians case
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
