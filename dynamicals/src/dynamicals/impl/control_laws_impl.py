from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from ..utils import Frame, position_error_clipped, rotation_error_clipped
from ..common.control_laws import AbstractControlLaw, ControlLawError


class CartesianVelocityControlLaw(AbstractControlLaw):
    """ Generates velocity commands in cartesian space with the law
                dx_tgt := dx_des + k * (x_des - x)
        where
            x               current pose (linear and angular)
            x_des, dx_des   desired pose and velocity (linear and angular)
            dx_tgt          target velocity (linear and angular)
    """

    def __init__(self, 
        k_p: float = 0., k_o: float = 0.,  
        min_actionable_error: Optional[ArrayLike] = None, 
        max_allowed_deviation: Optional[ArrayLike] = None
    ):
        # gains for the errors
        self.k_p : float
        self.k_o : float
        self.K : np.ndarray
        self.set_gains(k_p, k_o)
        # min target error to take action
        self.min_threshold : ArrayLike
        assert min_actionable_error is None or len(min_actionable_error) == 2, "min_threshold must be a 2-ndarray"
        self.set_min_threshold(min_actionable_error)
        # max deviation from current target
        self.max_deviation : ArrayLike
        assert max_allowed_deviation is None or len(max_allowed_deviation) == 2, "max_deviation must be a 2-ndarray"
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

    def set_min_threshold(self, min_threshold: ArrayLike):
        self.min_threshold = min_threshold
        
    def set_max_deviation(self, max_deviation: ArrayLike):
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
    
    def compute_target_state(self, raise_deviation: bool = True):
        """ Calculates the target velocities.
            :param raise_deviation: raise exception if max deviation is exceeded.
        """
        # manually handle `lower_bound` and `upper_bound` later using `min_threshold` and `max_deviation`
        error_pos_dir, error_pos_mag = position_error_clipped(self.current_position, self.desired_position, return_decomposed=True)
        error_rot_dir, error_rot_mag = rotation_error_clipped(self.current_rotation, self.desired_rotation, return_decomposed=True)
        
        # Check that the deviation from the trajectory is not too big.
        # This prevents absurdly high errors from being considered.
        if self.max_deviation is not None \
        and (error_pos_mag > self.max_deviation[0] or error_rot_mag > self.max_deviation[1]):
            if raise_deviation:
                raise ControlLawError("Deviation from current target too high")
            else:
                error_pos_mag = min(error_pos_mag, self.max_deviation[0])
                error_rot_mag = min(error_rot_mag, self.max_deviation[1])

        # Check that the error from the trajectory is not too small.
        # This prevents insignificant errors from being magnified by K gains.
        if self.min_threshold is not None:
            if error_pos_mag < self.min_threshold[0]:
                error_pos_mag = 0.
            if error_rot_mag < self.min_threshold[1]:
                error_rot_mag = 0.
            
        # update position and rotation (nothing to do here, this is just cache)
        self.target_position = self.desired_position
        self.target_rotation = self.desired_rotation
        # calculate velocity regardless of deviation
        self.target_velocity[0:3] = self.desired_velocity[0:3] + self.K[0:3] * error_pos_dir * error_pos_mag
        self.target_velocity[3:6] = self.desired_velocity[3:6] + self.K[3:6] * error_rot_dir * error_rot_mag
        
        return self.target_velocity
