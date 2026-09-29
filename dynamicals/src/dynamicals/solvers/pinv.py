from typing import Callable, Optional

import numpy as np
from numpy.typing import ArrayLike


def secondary_nothing(J, q, dq):
    return 0


class PINVSolver():

    def __init__(self, 
        J_size : ArrayLike,
        weights : Optional[ArrayLike] = None,
        damping : Optional[float] = None, 
        secondary_obj : Callable[[ArrayLike, ArrayLike], ArrayLike] = secondary_nothing
    ):
        """ :param weights: cost for each joint
        """
        self.secondary_obj = secondary_obj
        
        # weighted jacobian (W diagonal matrix)
        #   J+ = W^-1 J' (J W^-1 J')^-1
        self.W = np.asarray(weights)
        assert self.W.size == 1 or self.W.size == J_size[1], "Weights must be defined as a constant or consistent with given size"
        # performance can improved for static W using its Cholesky decomposition
        #   W^-1 = L L' = L^2  (L is diagonal as well)
        #   J+ = W^-1 J' (J W^-1 J')^-1
        #      = L (J L)' ((J L) (J L)')^-1
        self.L = np.sqrt(1/self.W)
        self._pinv_funct = self._pinv_w
        
        if damping is not None:
            # Tikhonov regularization
            # J+ = J' (J J' + G G')^-1
            # G = d I  (usually, making it L2 regularization)
            self.G = damping * np.eye(J_size[0])
            self.GGT = self.G @ self.G.T
            self._pinv_funct = self._pinv_w_d
        
        self._cached_eye_DOF = np.eye(J_size[1])
    
    def _pinv_w(self, J : np.ndarray):
        return self.L[:,None] * np.linalg.pinv(J * self.L)  # use * instead of @ for performance
    
    def _pinv_w_d(self, J : np.ndarray):
        JL = J * self.L
        return self.L[:,None] * JL.T @ np.linalg.inv(JL @ JL.T + self.GGT)
    
    def solve(self, v, J, q, dq):
        """ Solve the `v = J(q) @ dq` problem via least-squares error minimization.
            The function calculates `dq = J+ @ v + (I - J+ @ J) @ secondary`.
            :param v: desired output command
            :param J: current Jacobian
            :param q: current state positions
            :param dq: current state velocities
            :returns: command state velocities
        """
        jacobian_pinv = self._pinv_funct(J)
        ortho_proj = self._cached_eye_DOF - jacobian_pinv @ J
        dq_cmd = jacobian_pinv @ v + ortho_proj @ self.secondary_obj(J, q, dq)
        return dq_cmd
