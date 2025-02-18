from typing import Any
from abc import ABCMeta, abstractmethod

import numpy as np
import quaternion as quat

from .utils import Frame, normalize3, norm3
from .quat_utils import quat_diff


def position_error_clipped(current_pos: np.ndarray, target_pos: np.ndarray, max_error: float = np.inf, return_decomposed: bool = False):
    """ Calculates a clipped position error
    
        :param current_pos: np.array shape(3) [m]
        :param target_pos: np.array shape(3) [m]
        :param max_error: max error [m]
        :param return_decomposed: if true, return unitary direction and magnitude separately
    """
    position_error = (target_pos - current_pos)
    position_error_dir, position_error_dist = normalize3(position_error, return_norm=True)
    position_error_dist = min(max_error, position_error_dist)
    if return_decomposed:
        return position_error_dir, position_error_dist
    return position_error_dir * position_error_dist

def rotation_error_clipped(current_rot: np.quaternion, target_rot: np.quaternion, max_error: float = np.inf, return_decomposed: bool = False):
    """ Calculates a clipped angular error
    
        :param current_rot: quaternion np.array() shape(4)
        :param target_rot: quaternion np.array() shape(4)
        :param max_error: max error [rad]
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
    rotation_error_dir, rotation_error_angle = normalize3(2*np.log(rotation_error).vec, return_norm=True)
    rotation_error_angle = min(max_error, rotation_error_angle)
    if return_decomposed:
        return rotation_error_dir, rotation_error_angle
    return rotation_error_dir * rotation_error_angle


class ControlLawError(Exception):
    pass

class AbstractControlLaw(object, metaclass=ABCMeta):
    """ Abstract interface for a generic control law.
        Computation flow consists of the following steps:
        1. (optional) `update_current_timestep()`, which changes the value 
            of the time interval between current state and desired state
        2. `update_current_state()`, which modifies the internal initial state
        3. `update_desired_state()`, which modifies the internal final state
        4. `compute_target_state()`, which computes the required action to 
            achive the desired movement
        
        These steps are also available using `update_and_compute()`, which 
        calls all the required methods in the correct order. A `clear()` method 
        is available to clear the internal variables of the class.
    """
    
    def __init__(self, initial_timestep: float = 0.):
        super().__init__()
        self.dt = initial_timestep
    
    @abstractmethod
    def clear(self):
        """ Reset the internal variables of the control law
        """
        raise NotImplementedError()
    
    def update_current_timestep(self, timestep: float):
        """ Update the time internal between current and desired state
        """
        self.dt = timestep
    
    @abstractmethod
    def update_current_state(self, state: Any):
        """ Update the "now" state of the controlled system
        """
        raise NotImplementedError()
    
    @abstractmethod
    def update_desired_state(self, desired: Any):
        """ Update the "next" state of the controlled system
        """
        raise NotImplementedError()
    
    @abstractmethod
    def compute_target_state(self) -> Any:
        """ Compute the required action that brings the system from the current
            state to the desired state. If no action is found, this method can
            raise a `ControlLawError` exception.
        """
        raise NotImplementedError()

    def update_and_compute(self, current_state: Any, desired_state: Any, timestep: float) -> Any:
        """ Utility method that updates timestep, current and desired state and 
            immediately computes the required action. Useful when no extra logic 
            is required between the updates and the computation.
        """
        self.update_current_timestep(timestep)
        self.update_current_state(current_state)
        self.update_desired_state(desired_state)
        return self.compute_target_state()



# TODO kick me out
###############################################################################
###  IMPLEMENTATIONS
###############################################################################

class CartesianVelocityControlLaw(AbstractControlLaw):
    """ Generates velocity commands in cartesian space with the law
                dx_tgt := dx_des + k * (x_des - x)
        where
            x               current pose (linear and angular)
            x_des, dx_des   desired pose and velocity (linear and angular)
            dx_tgt          target velocity (linear and angular)
    """
    
    def __init__(self, k_p: float = 0., k_o: float = 0., max_deviation: np.ndarray = None):
        # gains for the errors
        self.k_p : float
        self.k_o : float
        self.K : np.ndarray
        self.set_gains(k_p, k_o)
        # max deviation from current target
        self.max_deviation : np.ndarray
        assert max_deviation.size == 2, "max_deviation must be a 2-ndarray"
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

    # TODO second look here
    def clear(self):
        self.target_velocity = np.zeros(6)

    def _update_gains(self):
        self.K = np.array([self.k_p, self.k_p, self.k_p, self.k_o, self.k_o, self.k_o])

    def set_gains(self, k_p: float, k_o: float):
        self.k_p = k_p
        self.k_o = k_o
        self._update_gains()

    def set_position_gain(self, gain: float):
        self.set_gains(gain, self.k_o)

    def set_rotation_gain(self, gain: float):
        self.set_gains(self.k_p, gain)

    def set_max_deviation(self, max_deviation: np.ndarray):
        self.max_deviation = max_deviation

    def update_current_state(self, current_pose: Frame):
        """ Updates the pose and calculates the pose for relative and absolute motion as well
        """
        self.current_position = current_pose.pos
        self.current_rotation = current_pose.rot
        self.current_velocity = current_pose.vel

    def update_desired_state(self, desired_pose: Frame):
        self.desired_position = desired_pose.pos
        self.desired_rotation = desired_pose.rot
        self.desired_velocity = desired_pose.vel

    def _check_deviation(self, error: np.ndarray):
        """ Returns true if any of the deviation limits for target following has been violated.
            :param max_deviation: np.array([max_position_deviation, max_rotation_deviation]), shape(2)
        """
        error_position = norm3(error[0:3])
        error_rotation = norm3(error[3:6])
        violated = error_position > self.max_deviation[0] or error_rotation > self.max_deviation[1]
        return violated

    def compute_target_state(self, raise_deviation: bool = True):
        """ Calculates the target velocities.
            :param raise_deviation: raise exception if max deviation is exceeded.
        """
        error_pos_dir, error_pos_mag = position_error_clipped(self.current_position, self.desired_position, 0.1, True)
        error_rot_dir, error_rot_mag = rotation_error_clipped(self.current_rotation, self.desired_rotation, 0.2, True)

        # update position and rotation (nothing to do here)
        self.target_position = self.desired_position
        self.target_rotation = self.desired_rotation
        # calculate velocity regardless of deviation
        # TODO slicing arrays is apparently slow...
        self.target_velocity[0] = self.desired_velocity[0] + self.K[0] * error_pos_dir[0] * error_pos_mag
        self.target_velocity[1] = self.desired_velocity[1] + self.K[1] * error_pos_dir[1] * error_pos_mag
        self.target_velocity[2] = self.desired_velocity[2] + self.K[2] * error_pos_dir[2] * error_pos_mag
        self.target_velocity[3] = self.desired_velocity[3] + self.K[3] * error_rot_dir[0] * error_rot_mag
        self.target_velocity[4] = self.desired_velocity[4] + self.K[4] * error_rot_dir[1] * error_rot_mag
        self.target_velocity[5] = self.desired_velocity[5] + self.K[5] * error_rot_dir[2] * error_rot_mag

        # check that the deviation from the trajectory is not too big
        # here we could also call `self._check_deviation()`, but re-computing 
        # the norm is not necessary since `*_error_clipped()` methods can return
        # direction and magnitudes separately
        if raise_deviation and self.max_deviation is not None \
        and (error_pos_mag > self.max_deviation[0] or error_rot_mag > self.max_deviation[1]):
            raise ControlLawError("Deviation from current target too high")

        return self.target_velocity
