from typing import Tuple

import numpy as np
import quaternion as quat

from . import quat_utils

###############################################################################
#                         TUSTIN-DISCRETIZED MODELS                           #
###############################################################################

class AdmittanceTustin(object):
    
    def __init__(self, m, k, d, h) -> None:
        """ Create a n-dof dimensional admittance
            :param m: mass of the admittance (float or ndarray)
            :param k: spring of the admittance (float or ndarray)
            :param d: damping of the admittance (float, ndarray, or None for critically damped system)
            :param h: step size (float)
        """
        m = np.asarray(m)
        k = np.asarray(k)
        d = 2*np.sqrt(m*k) if d is None else np.asarray(d)
        try:
            m, k, d = np.broadcast_arrays(m, k, d)
            if m.ndim > 1:
                raise Exception
        except Exception:
            raise ValueError("m, d, and k must be either floats or 1-d arrays with the same length")
        if not isinstance(h, float):
            raise ValueError("h must be float")
        self.m = m
        self.k = k
        self.d = d
        self.h = 0
        self.n = m.size
        self._setup_coeffs(h)
        # initial window for both the input u and the output y
        self._u_1 = 0
        self._u_2 = 0
        self._y_1 = 0
        self._y_2 = 0
        # last calculated values
        self.y = 0
        self.dy = 0
        self.ddy = 0
        
    def _setup_coeffs(self, h) -> None:
        self.h = h
        h2 = 2/self.h
        h2_2 = h2**2
        A0 = self.m * h2_2 + self.d * h2 + self.k
        A1 = -2 * self.m * h2_2 + 2 * self.k
        A2 = self.m * h2_2 - self.d * h2 + self.k
        B0 = 1
        B1 = 2
        B2 = 1
        DB0 = h2
        DB1 = 0
        DB2 = -h2
        DDB0 = h2_2
        DDB1 = -2 * h2_2
        DDB2 = h2_2
        # actual coefficient used for computation
        self.D0 = A1 / A0
        self.D1 = A2 / A0
        self.C0 = B0 / A0
        self.C1 = B1 / A0
        self.C2 = B2 / A0
        self.DC0 = DB0 / A0
        self.DC1 = DB1 / A0  # useless, but the overhead is ignorable
        self.DC2 = DB2 / A0
        self.DDC0 = DDB0 / A0
        self.DDC1 = DDB1 / A0
        self.DDC2 = DDB2 / A0
    
    def __call__(self, u) -> float:
        """ Returns position given an input force.
        """
        y, _, _ = self.compute(u)
        return y
    
    def compute(self, u, h=None) -> Tuple[float, float, float]:
        """ Returns position, velocity and acceleration given an input force.
        """
        if h is not None:
            self._setup_coeffs(h)
        e = self.D0 * self._y_1 + self.D1 * self._y_2
        y   = self.C0 * u + self.C1 * self._u_1 + self.C2 * self._u_2 - e
        dy  = self.DC0 * u + self.DC1 * self._u_1 + self.DC2 * self._u_2 - e
        ddy = self.DDC0 * u + self.DDC1 * self._u_1 + self.DDC2 * self._u_2 - e
        self._update_window(u, y)
        self._update_output(y, dy, ddy)
        return y, dy, ddy
    
    def _update_window(self, u, y) -> None:
        self._u_2 = self._u_1
        self._u_1 = u
        self._y_2 = self._y_1
        self._y_1 = y

    def _update_output(self, y, dy, ddy) -> None:
        self.y = y
        self.dy = dy
        self.ddy = ddy


class LPFilterTustin(object):
    
    def __init__(self, f, k, h) -> None:
        """ Create a n-dof dimensional admittance
            :param f: cutoff frequency (float or ndarray)
            :param k: static gain (float or ndarray)
            :param h: step size (float)
        """
        w = 2*np.pi*np.asarray(f)
        k = np.asarray(k)
        try:
            w, k = np.broadcast_arrays(w, k)
            if w.ndim > 1:
                raise Exception
        except Exception:
            raise ValueError("w and k must be either floats or 1-d arrays with the same length")
        if not isinstance(h, float):
            raise ValueError("h must be float")
        self.w = w
        self.tau = 1/self.w
        self.k = k
        self.h = h
        self.n = w.size
        self._setup_coeffs()
        # initial window for both the input u and the output y
        self._u_1 = 0
        self._y_1 = 0
        # last calculated values
        self.y = 0
        
    def _setup_coeffs(self) -> None:
        h_ = 2/self.h
        A0 = 1 + h_ * self.tau
        A1 = 1 - h_ * self.tau
        B0 = self.k
        B1 = self.k
        # actual coefficient used for computation
        self.D0 = A1 / A0
        self.C0 = B0 / A0
        self.C1 = B1 / A0
    
    def __call__(self, u) -> float:
        """ Returns position given an input force.
        """
        y = self.compute(u)
        return y
    
    def compute(self, u) -> float:
        """ Returns position, velocity and acceleration given an input force.
        """
        e = self.D0 * self._y_1
        y = self.C0 * u + self.C1 * self._u_1 - e
        self._update_window(u, y)
        self._update_output(y)
        return y
    
    def _update_window(self, u, y) -> None:
        self._u_1 = u
        self._y_1 = y

    def _update_output(self, y) -> None:
        self.y = y


###############################################################################
#                             STATE SPACE MODELS                              #
###############################################################################

class DiscretizedStateSpaceModel(object):
    """ Implementation of a discretized time-invariant state-space model.
        Discretization can be either forward/backward Euler or Tustin.
        More at https://en.wikipedia.org/wiki/Discretization
    """
    
    def __init__(self, 
        A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, h: float, 
        x0: np.ndarray = None, 
        method="forward"
    ) -> None:
        """ Create a n-dof dimensional state-space model
            :param A: state commutation matrix
            :param B: input commutation matrix
            :param C: state-output matrix (can be None)
            :param D: input-output matrix (can be None)
            :param h: step size
            :param x0: initial state of the system (assumed zeros if None)
            :param method: approximation method {exact, forward, backward, tustin}
        """
        ALLOWED_METHODS = ["exact", "forward", "backward", "tustin"]
        
        # check method
        assert method in ALLOWED_METHODS, f"method must be one of {{ {', '.join(ALLOWED_METHODS)} }}"
        self.method = method
        
        # check A dimensions
        assert A.ndim == 2, "A must be 2d"
        assert A.shape[0] == A.shape[1], "A must be a square matrix"
        # check B dimensions
        assert 0 <= B.ndim and B.ndim <= 2, "B must be either 1d or 2d"
        if B.ndim == 1:
            B = B[:, np.newaxis]
        assert A.shape[0] == B.shape[0], "A and B must have same number of rows (check B)"
        # check C and D dimensions
        assert (C is None) == (D is None), "Both C and D must be either set or None"
        if C is not None:  # and D is not None
            # check C dimenions
            assert C.ndim == 2, "C must be 2d"
            assert A.shape[1] == C.shape[1], "A and C must have same number of columns (check C)"
            # check D dimenions
            D = np.array(D)
            assert 0 <= D.ndim and D.ndim <= 2, "D must be either 1d or 2d"
            if D.ndim == 0:
                D = D[np.newaxis, np.newaxis]
            if D.ndim == 1:
                D = D[:, np.newaxis]
            assert D.shape[0] == C.shape[0], "C and D must have same number of rows (check D)"
            assert D.shape[1] == B.shape[1], "B and D must have same number of columns (check D)"
        
        # initial state
        if x0 is not None:
            assert 1 <= x0.ndim and x0.ndim <= 2, "Initial state must be 1d (for single evaluation) or 2d (for multiple evaluation)"
            assert x0.shape[0] == A.shape[0], "Initial state must have same size as A"
        else:
            x0 = np.zeros((A.shape[0],))
        
        # state, input, and output dimensions, and number of systems evaluated simultaneously
        self.n, self.m, self.p = A.shape[0], B.shape[1], C.shape[0] if C is not None else 0
        # system matrices and vectors
        self.A, self.B, self.C, self.D, self.x0 = A, B, C, D, x0
                
        # cache to speed up evolution calculation
        self._cache_G = np.zeros((self.n + self.p, self.n + self.m))
        if C is not None:
            self._cache_G[self.n:, :self.n] = C
            self._cache_G[self.n:, :self.n] = D
        self._cache_X = np.zeros((self.n + self.m,))  # [x, u_new]
        self._cache_Y = np.zeros((self.n + self.p,))  # [x_new, y_new]
        self._eye_n = np.eye(self.n)
        # finally, populate cache
        self._setup_coeffs(h)
        self.reset()
        
        # alias for internal state
        self.x = self._cache_Y[:self.n]
        self.y = self._cache_Y[self.n:]
        
    def _setup_coeffs(self, h: float) -> None:
        self.h = h
        # actual coefficient used for computation
        if self.method == "exact":
            # TODO here we suppose `A` is diagonalizable, add jordanization
            L, V = np.linalg.eig(self.A)  # A = V @ diag(L)*h @ inv(V)
            eAh = V @ np.exp(np.diag(L*h)) @ np.linalg.inv(V)
            G = np.linalg.inv(self.A) @ (eAh - self._eye_n) @ self.B
            # FIXME wtf happened here?
            raise RuntimeError("exact method is currently broken")
        elif self.method == "forward":
            eAh = self._eye_n + self.A * h
            G = self.B * h
        elif self.method == "backward":
            eAh = np.linalg.inv(self._eye_n - self.A * h)
            G = eAh @ self.B * h
        elif self.method == "tustin":
            eAh = (self._eye_n + 0.5 * self.A * h) @ np.linalg.inv(self._eye_n - 0.5 * self.A * h)
            G = np.linalg.inv(self.A) @ (eAh - self._eye_n) @ self.B
        else:
            raise RuntimeError("no such method found")
        
        self._cache_G[:self.n, :self.n] = eAh
        self._cache_G[:self.n, self.n:] = G
    
    def reset(self):
        self._cache_Y[:self.n] = self.x0  # set last output state as first input state
    
    def __call__(self, u: np.ndarray, h_new: float = None):
        """ Returns the state of the system for a given input.
            Override this method if output from `__call__` must be manipulated.
            :param u: input for the system with `shape(self.m)` or `shape(self.m,self.s)`
            :param h_new: timestep for the new input
        """
        return self.compute(u, h_new, False)
    
    def compute(self, u: np.ndarray, h_new: float = None, return_output: bool = False):
        """ Returns the state (and the output) of the system for a given input.
            :param u: `np.ndarray` with `shape(self.m)`
            :param h_new: timestep for the new input
            :param return_output: wether to return also the `y` output or just the `x` state of the system
        """
        if h_new is not None:
            self._setup_coeffs(h_new)
        # compute the new state
        self._cache_X[:self.n] = self._cache_Y[:self.n]  # use last output state as new input state
        self._cache_X[self.n:] = u
        self._cache_Y[:] = self._cache_G @ self._cache_X
        # compute the new output
        if return_output:
            return self.x, self.y
        else:
            return self.x

    def compute_signal(self, U: np.ndarray):
        """ Compute the system over the provided time-series
            :param U: time-series with `shape(?,self.m)`
        """
        T = U.shape[0]
        X = np.zeros((T, self.n))
        for t in range(T):
            X[t, ...] = self.compute(U[t, ...])
        return X


class Admittance(DiscretizedStateSpaceModel):
    
    def __init__(self, M, K, D, h, n=None, method="forward") -> None:
        """ Create a n-dof dimensional admittance
            :param m: mass of the admittance (float, 1-d, or 2-d ndarray)
            :param k: spring of the admittance (float, 1-d, or 2-d ndarray)
            :param d: damping of the admittance (float, 1-d, 2-d ndarray, or None for critically damped system)
            :param h: step size (float or 1-d ndarray)
            :param n: size of the input (int or None)
            :param method: approximation method {exact, forward, backward, tustin}
        """
        def reshape(v, n):
            v: np.ndarray = np.asarray(v)
            if v.ndim == 0:
                v = v * np.eye(n)
            elif v.ndim == 1 and v.shape == (n,):
                v = np.diag(v)
            elif v.ndim == 2 and v.shape == (n,n):
                pass
            else:
                raise ValueError(f"shape {v.shape} is not consistent with size n={n}")
            return v

        def matrix_sqrt(V):
            # Computing diagonalization
            E, V = np.linalg.eig(V)  # TODO this assumes V is diagonalizable
            # Ensuring square root matrix exists
            assert np.all(E >= 0)
            sqrt_matrix = V * np.sqrt(E) @ np.linalg.inv(V)
            return sqrt_matrix
        
        if n is None:
            # the first tuple is to ensure at least one dimension
            n = np.max(np.concatenate([(1,), np.shape(M), np.shape(K), np.shape(D) if D is not None else ()])).astype(int)
        
        # store admittance parameters
        self.M = reshape(M, n)
        self.K = reshape(K, n)
        self.D = reshape(D, n) if D is not None else 2*matrix_sqrt((self.M @ self.K))
        self.dims = n
        
        # prepare blocks for linear system
        invM = np.linalg.inv(self.M)
        A = np.block([[np.zeros((n,n)),      np.eye(n)], 
                      [ -invM @ self.K, -invM @ self.D]])
        B = np.block([[np.zeros((n,n))], 
                      [invM]])
        
        super().__init__(A, B, None, None, h, None, method)
    
    def compute(self, u: np.ndarray, h_new: float = None):  
        """ Returns "position" and "velocity" for given input.
        """
        x = super().compute(u, h_new)
        y, dy = x[:self.dims], x[self.dims:]
        return y, dy
    
    def compute_signal(self, U: np.ndarray):
        T = U.shape[0]
        X = np.zeros((T, self.dims))
        dX = np.zeros((T, self.dims))
        for t in range(T):
            X[t, ...], dX[t, ...] = self.compute(U[t, ...])
        return X, dX

class AdmittanceForce(Admittance):
    def __init__(self, M, K, D, h, method="forward") -> None:
        super().__init__(M, K, D, h, 3, method)
    
    def compute(self, f: np.ndarray, h_new: float = None):    
        """ Returns position and velocity given an input force.
            This function can run at minimum 14kHz in "forward" mode on a decent laptop.
        """
        return super().compute(f, h_new)

class AdmittanceTorque(Admittance):
    
    def __init__(self, M, K, D, h, method="forward") -> None:
        super().__init__(M, K, D, h, 3, method)
    
    def compute(self, m: np.ndarray, h_new: float = None):    
        """ Returns rotation and angular velocity given an input torque.
            This function can run at minimum 13kHz in "forward" mode on a decent laptop.
        """
        # q = log(Q), dq is its derivative
        # Q is the rotation quaternion, W (omega, the angular velocity) is its derivative
        q, dq = super().compute(m, h_new)
        Q = np.exp(quat.quaternion(0, *q) / 2)  # `quat.from_rotation_vector` is slow, do it manually
        w = 2*quat.quaternion(*(quat_utils.jac_q(q) @ dq)) * Q.conj()
        W = w.vec
        return Q, W

    def compute_signal(self, U: np.ndarray):
        T = U.shape[0]
        X = np.zeros((T,), dtype=np.quaternion)
        dX = np.zeros((T, self.dims))
        for t in range(T):
            X[t, ...], dX[t, ...] = self.compute(U[t, ...])
        return X, dX
    
class AdmittanceWrench(Admittance):
    def __init__(self, M, K, D, h, method="forward") -> None:
        super().__init__(M, K, D, h, 6, method)
    
    def compute(self, w: np.ndarray, h_new: float = None):    
        """ Returns position and velocity given an input wrench.
            This function can run at minimum 6kHz in "forward" mode on a decent laptop.
        """
        x, dx = super().compute(w, h_new)
        p, dp, q, dq = x[:3], dx[:3], x[3:], dx[3:]
        Q = np.exp(quat.quaternion(0, *q) / 2)  # `quat.from_rotation_vector` is slow, do it manually
        w = 2*quat.quaternion(*(quat_utils.jac_q(q) @ dq)) * Q.conj()
        W = w.vec
        return (p, Q), (dp, W)

    def compute_signal(self, U: np.ndarray):
        T = U.shape[0]
        P = np.zeros((T, self.dims))
        dP = np.zeros((T, self.dims))
        Q = np.zeros((T,), dtype=np.quaternion)
        W = np.zeros((T, self.dims))
        for t in range(T):
            (P[t, ...], Q[t, ...]), (dP[t, ...], W[t, ...]) = self.compute(U[t, ...])
        return (P, Q), (dP, W)