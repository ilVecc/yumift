import numpy as np
import quaternion as quat

from ..utils import normalize3
from ..utils.quaternions import quat_diff


def vector_clipped(vector : np.ndarray, lb: float = 0, ub: float = np.inf, return_decomposed: bool = False):
    """ Calculates a clipped norm vector
    
        :param vector: vector to be clipped
        :param lb: norm lower bound
        :param ub: norm upper bound
        :param return_decomposed: if true, return unitary direction and magnitude separately
    """
    dir, mag = normalize3(vector, return_norm=True)
    mag = max(lb, min(ub, mag))
    if return_decomposed:
        return dir, mag
    return dir * mag


def position_error_clipped(
    current_pos: np.ndarray, target_pos: np.ndarray, 
    lower_bound: float = 0, upper_bound: float = np.inf, 
    return_decomposed: bool = False
):
    """ Calculates a clipped position error
    
        :param current_pos: np.array shape(3) [m]
        :param target_pos: np.array shape(3) [m]
        :param lower_bound: min error [m]
        :param upper_bound: max error [m]
        :param return_decomposed: if true, return unitary direction and magnitude separately
    """
    position_error = (target_pos - current_pos)
    return vector_clipped(position_error, lower_bound, upper_bound, return_decomposed)


def rotation_error_clipped(
    current_rot: np.quaternion, target_rot: np.quaternion, 
    lower_bound: float = 0, upper_bound: float = np.inf, 
    return_decomposed: bool = False
):
    """ Calculates a clipped angular error
    
        :param current_rot: quaternion np.array() shape(4)
        :param target_rot: quaternion np.array() shape(4)
        :param lower_bound: min error [rad]
        :param upper_bound: max error [rad]
        :param return_decomposed: if true, return unitary direction and magnitude separately
    """
    # In Siciliano, the rotation error is `eo = (qf * inv(qi)).vec` since bringing 
    # it to [0 0 0] means obtaining the relative quaternion `qf * inv(qi) = {1,[0 0 0]}`.
    # Dealing with the scalar part is thus redundant and can be omitted. 
    # Here, since we allow to set a maximum error, we have to explicitly deal with it.
    rotation_diff = quat_diff(current_rot, target_rot)
    # here `quat.as_rotation_vector(rotation_diff)` could be used, but it's 
    # meant for collections of quaternions, thus expensive, so we use the 
    # underlying operations directly, assuming `rotation_diff` is normalized
    rotation_error = 2*np.log(rotation_diff).vec
    return vector_clipped(rotation_error, lower_bound, upper_bound, return_decomposed)
