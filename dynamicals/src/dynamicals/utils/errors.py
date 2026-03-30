import numpy as np
import quaternion as quat

from ..utils import normalize3
from ..utils.quaternions import quat_diff


def position_error_clipped(current_pos: np.ndarray, target_pos: np.ndarray, lower_bound: float = 0, upper_bound: float = np.inf, return_decomposed: bool = False):
    """ Calculates a clipped position error
    
        :param current_pos: np.array shape(3) [m]
        :param target_pos: np.array shape(3) [m]
        :param lower_bound: min error [m]
        :param upper_bound: max error [m]
        :param return_decomposed: if true, return unitary direction and magnitude separately
    """
    position_error = (target_pos - current_pos)
    position_error_dir, position_error_mag = normalize3(position_error, return_norm=True)
    position_error_mag = max(lower_bound, min(upper_bound, position_error_mag))
    if return_decomposed:
        return position_error_dir, position_error_mag
    return position_error_dir * position_error_mag


def rotation_error_clipped(current_rot: np.quaternion, target_rot: np.quaternion, lower_bound: float = 0, upper_bound: float = np.inf, return_decomposed: bool = False):
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
    rotation_error = quat_diff(current_rot, target_rot)
    # here `quat.as_rotation_vector(rotation_error)` could be used, but it's 
    # meant for collections of quaternions, thus expensive, so we use the 
    # underlying operations directly, assuming `rotation_error` is normalized
    rotation_error_dir, rotation_error_mag = normalize3(2*np.log(rotation_error).vec, return_norm=True)
    rotation_error_mag = max(lower_bound, min(upper_bound, rotation_error_mag))
    if return_decomposed:
        return rotation_error_dir, rotation_error_mag
    return rotation_error_dir * rotation_error_mag

