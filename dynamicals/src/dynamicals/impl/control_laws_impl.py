import numpy as np

from ..utils import Frame, norm3, position_error_clipped, rotation_error_clipped
from ..common.control_laws import AbstractControlLaw, ControlLawError


class CartesianVelocityControlLaw(AbstractControlLaw):
    """ Generates velocity commands in cartesian space with the law
                dx_tgt := dx_des + k * (x_des - x)
        where
            x               current pose (linear and angular)
            x_des, dx_des   desired pose and velocity (linear and angular)
            dx_tgt          target velocity (linear and angular)
    """

    def __init__(self, k_p: float = 0., k_o: float = 0.,  min_actionable_error: np.ndarray = None, max_allowed_deviation: np.ndarray = None):
        # gains for the errors
        self.k_p : float
        self.k_o : float
        self.K : np.ndarray
        self.set_gains(k_p, k_o)
        # min target error to take action
        self.min_error : np.ndarray
        assert len(min_actionable_error) == 2, "min_error must be a 2-ndarray"
        self.set_min_error(min_actionable_error)
        # max deviation from current target
        self.max_deviation : np.ndarray
        assert len(max_allowed_deviation) == 2, "max_deviation must be a 2-ndarray"
        self.set_max_deviation(max_allowed_deviation)

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

    def set_min_error(self, min_error: np.ndarray):
        self.min_error = min_error
        
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

    #deprecated
    def _check_error(self, error: np.ndarray):
        """ Returns true if any of the target error is lower than the minimum required.
            :param min_error: np.array([min_position_error, min_rotation_error]), shape(2)
        """
        error_position, error_rotation = norm3(error[0:3]), norm3(error[3:6])
        insufficient = error_position < self.min_error[0] and error_rotation < self.min_error[1]
        return insufficient

    #deprecated
    def _check_deviation(self, error: np.ndarray):
        """ Returns true if any of the deviation limits for target following has been violated.
            :param max_deviation: np.array([max_position_deviation, max_rotation_deviation]), shape(2)
        """
        error_position, error_rotation = norm3(error[0:3]), norm3(error[3:6])
        violated = error_position > self.max_deviation[0] or error_rotation > self.max_deviation[1]
        return violated

    def compute_target_state(self, raise_deviation: bool = True):
        """ Calculates the target velocities.
            :param raise_deviation: raise exception if max deviation is exceeded.
        """
        # HACK lower_bound can be too high for the min_error to trigger, make this a parameter or use min_error itself
        # HACK upper_bound can be too low for the max_deviation to trigger, make this a parameter or use max_deviation itself
        error_pos_dir, error_pos_mag = position_error_clipped(self.current_position, self.desired_position, lower_bound=0, upper_bound=1.5, return_decomposed=True)
        error_rot_dir, error_rot_mag = rotation_error_clipped(self.current_rotation, self.desired_rotation, lower_bound=0, upper_bound=1.5, return_decomposed=True)

        # update position and rotation (nothing to do here)
        self.target_position = self.desired_position
        self.target_rotation = self.desired_rotation
        
        # check that the error from the trajectory is not too small
        # here we could also call `self._check_error()`, but re-computing 
        # the norm is not necessary since `*_error_clipped()` methods can return
        # direction and magnitudes separately
        if self.min_error is not None \
        and (error_pos_mag < self.min_error[0] and error_rot_mag < self.min_error[1]):
            self.target_velocity[:] = 0
        else:
            # calculate velocity regardless of deviation
            self.target_velocity[0:3] = self.desired_velocity[0:3] + self.K[0:3] * error_pos_dir * error_pos_mag
            self.target_velocity[3:6] = self.desired_velocity[3:6] + self.K[3:6] * error_rot_dir * error_rot_mag
        
        # check that the deviation from the trajectory is not too big
        # here we could also call `self._check_deviation()`, but re-computing 
        # the norm is not necessary since `*_error_clipped()` methods can return
        # direction and magnitudes separately
        if raise_deviation and self.max_deviation is not None \
        and (error_pos_mag > self.max_deviation[0] or error_rot_mag > self.max_deviation[1]):
            raise ControlLawError("Deviation from current target too high")
        
        return self.target_velocity
