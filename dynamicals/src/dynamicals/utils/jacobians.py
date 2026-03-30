from typing import Optional

import numpy as np
import quaternion as quat

from .geometry import skew_matrix

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
