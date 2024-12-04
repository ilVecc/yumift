import numpy as np
import quaternion as quat

from . import utils
from .quat_utils import quat_diff

def position_error_clipped(current_pos: np.ndarray, target_pos: np.ndarray, max_error=np.inf):
    """ Calculates a clipped position error
        :param current_pos: np.array shape(3) [m]
        :param target_pos: np.array shape(3) [m]
        :param max_error: max error [m]
    """
    position_error = (target_pos - current_pos)
    position_error_dir, position_error_dist = utils.normalize(position_error, return_norm=True)
    return position_error_dir * min([max_error, position_error_dist])


def rotation_error_clipped(current_rot: np.quaternion, target_rot: np.quaternion, max_error=np.inf):
    """ Calculates a clipped angular error
        :param current_rot: quaternion np.array() shape(4)
        :param target_rot: quaternion np.array() shape(4)
        :param max_error: max error [rad]
    """
    rotation_error = quat_diff(current_rot, target_rot)
    rotation_error_dir, rotation_error_angle = utils.normalize(quat.as_rotation_vector(rotation_error), return_norm=True)
    return rotation_error_dir * min([max_error, rotation_error_angle])


class ControlLawError(Exception):
    pass


class CartesianVelocityControlLaw(object):
    """ Generates velocity commands in cartesian space for following a trajectory
    """
    def __init__(self, k_p=0, k_o=0, max_deviation=None):
        # gains for the errors
        self.k_p : float
        self.k_o : float
        self.K : np.ndarray
        self.set_gains(k_p, k_o)
        # max deviation from current target
        self.max_deviation : np.ndarray
        self.set_max_deviation(max_deviation)

        ### position/rotation/velocity variables 
        ### (in [m], [rad], and a mixture of [m/s] and [rad/s])
        # state variables (where we are now)
        self.current_position : np.ndarray  # shape(3)
        self.current_rotation : np.quaternion
        self.current_velocity : np.ndarray  # shape(6)
        # desired variables (what we want to obtain)
        self.desired_position : np.ndarray  # shape(3)
        self.desired_rotation : np.quaternion
        self.desired_velocity : np.ndarray  # shape(6)
        # target variables (what to send to the controller)
        self.target_position : np.ndarray  # shape(3)
        self.target_rotation : np.quaternion
        self.target_velocity : np.ndarray  # shape(6)
        self.clear()

    def clear(self):
        self.target_velocity = np.zeros(6)

    def _update_gains(self):
        self.K = np.array([self.k_p, self.k_p, self.k_p, self.k_o, self.k_o, self.k_o])

    def set_gains(self, k_p, k_o):
        self.k_p = k_p
        self.k_o = k_o
        self._update_gains()

    def set_position_gain(self, gain):
        self.set_gains(gain, self.k_o)

    def set_rotation_gain(self, gain):
        self.set_gains(self.k_p, gain)

    def set_max_deviation(self, max_deviation):
        self.max_deviation = max_deviation

    def update_current_pose(self, current_pose: utils.Frame):
        """ Updates the pose and calculates the pose for relative and absolute motion as well
        """
        self.current_position = current_pose.pos
        self.current_rotation = current_pose.rot
        self.current_velocity = current_pose.vel

    def update_desired_pose(self, desired_pose: utils.Frame):
        self.desired_position = desired_pose.pos
        self.desired_rotation = desired_pose.rot
        self.desired_velocity = desired_pose.vel

    def _check_deviation(self, error):
        """ Returns true if any of the deviation limits for target following has been violated.
            :param max_deviation: np.array([max_position_deviation, max_rotation_deviation]), shape(2)
        """
        error_position = np.linalg.norm(error[0:3])
        error_rotation = np.linalg.norm(error[3:6])
        errors = np.array([error_position, error_rotation])
        violated = np.any(errors > self.max_deviation)
        return violated

    def compute_target_velocity(self, raise_deviation=True):
        """ Calculates the target velocities.
            :param k_p: float for position gain.
            :param k_o: float for orientation gain.
        """
        error = np.concatenate([
            position_error_clipped(self.current_position, self.desired_position, 0.1),
            rotation_error_clipped(self.current_rotation, self.desired_rotation, 0.2)
        ])

        # update position and rotation (nothing to do here)
        self.target_position = self.desired_position
        self.target_rotation = self.desired_rotation
        # calculate velocity regardless of deviation
        self.target_velocity = self.desired_velocity + self.K * error

        # check that the deviation from the trajectory is not too big
        if raise_deviation and self.max_deviation is not None and self._check_deviation(error):
            raise ControlLawError("Deviation from current target too high")

        return self.target_velocity
