from enum import Enum
import numpy as np
import quaternion as quat

class DiscretizationMethod(Enum):
    EXACT = "exact"
    FORWARD = "forward"
    BACKWARD = "backward"
    TUSTIN = "tustin"
    
    @staticmethod
    def from_str(value : str):
        return DiscretizationMethod[value.upper()]
        
class DiscretizedStateSpaceModel(object):
    """ Implementation of a discretized time-invariant state-space model.
        Discretization can be either forward/backward Euler or Tustin.
        More at https://en.wikipedia.org/wiki/Discretization
    """
    
    def __init__(self, 
        A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, h: float, 
        x0: np.ndarray = None, 
        method: DiscretizationMethod = DiscretizationMethod.FORWARD
    ) -> None:
        """ Create a n-dof dimensional state-space model
            :param A: state commutation matrix
            :param B: input commutation matrix
            :param C: state-output matrix (can be None)
            :param D: input-output matrix (can be None)
            :param h: step size
            :param x0: initial state of the system (assumed zeros if None)
            :param method: approximation method
        """
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
        if self.method == DiscretizationMethod.EXACT:
            # TODO here we suppose `A` is diagonalizable, add jordanization
            L, V = np.linalg.eig(self.A)  # A = V @ diag(L)*h @ inv(V)
            eAh = V @ np.exp(np.diag(L*h)) @ np.linalg.inv(V)
            G = np.linalg.inv(self.A) @ (eAh - self._eye_n) @ self.B
            # FIXME wtf happened here?
            raise RuntimeError("exact method is currently broken")
        elif self.method == DiscretizationMethod.FORWARD:
            eAh = self._eye_n + self.A * h
            G = self.B * h
        elif self.method == DiscretizationMethod.BACKWARD:
            eAh = np.linalg.inv(self._eye_n - self.A * h)
            G = eAh @ self.B * h
        elif self.method == DiscretizationMethod.TUSTIN:
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


class LPFilter(DiscretizedStateSpaceModel):
    
    def __init__(self, freq, gain=1, n=1, h=0.001, method=DiscretizationMethod.FORWARD) -> None:
        A = -freq * np.eye(n)
        B = gain * freq * np.eye(n)
        super().__init__(A, B, None, None, h, None, method)


class Admittance(DiscretizedStateSpaceModel):
    
    def __init__(self, M, D, K, h, n=None, method=DiscretizationMethod.FORWARD) -> None:
        """ Create a n-dof dimensional admittance
            :param m: mass of the admittance (float, 1-d, or 2-d ndarray)
            :param d: damping of the admittance (float, 1-d, 2-d ndarray, or None for critically damped system)
            :param k: spring of the admittance (float, 1-d, or 2-d ndarray)
            :param h: step size (float or 1-d ndarray)
            :param n: size of the input (int or None)
            :param method: approximation method {exact, forward, backward, tustin}
        """
        if n is None:
            # the first tuple is to ensure at least one dimension
            n = np.max(np.concatenate([(1,), np.shape(M), np.shape(K), np.shape(D) if D is not None else ()])).astype(int)
        
        # store admittance parameters
        self.M = self._reshape(M, n)
        self.K = self._reshape(K, n)
        self.D = self._reshape(D, n) if D is not None else np.diag([None]*n)
        if np.any(self.D == None):
            self.D[self.D == None] = 2*self._matrix_sqrt((self.M[self.D == None] @ self.K[self.D == None]))
        self.D = self.D.astype(float)
        self.dims = n
        
        # prepare blocks for linear system
        invM = np.linalg.inv(self.M)
        A = np.block([[np.zeros((n,n)),      np.eye(n)], 
                      [ -invM @ self.K, -invM @ self.D]])
        B = np.block([[np.zeros((n,n))], 
                      [invM]])
        
        super().__init__(A, B, None, None, h, None, method)
    
    @staticmethod
    def _reshape(v, n):
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

    @staticmethod
    def _matrix_sqrt(V):
        # Computing diagonalization
        E, V = np.linalg.eig(V)  # TODO this assumes V is diagonalizable
        # Ensuring square root matrix exists
        assert np.all(E >= 0)
        return V * np.sqrt(E) @ np.linalg.inv(V)
    
    def compute(self, u: np.ndarray, h_new: float = None):  
        """ Returns "position" and "velocity" for given input.
        """
        x = super().compute(u, h_new)
        return x[:self.dims], x[self.dims:]  # y, dy
    
    def compute_signal(self, U: np.ndarray):
        T = U.shape[0]
        X = np.zeros((T, self.dims))
        dX = np.zeros((T, self.dims))
        for t in range(T):
            X[t, ...], dX[t, ...] = self.compute(U[t, ...])
        return X, dX

class AdmittanceForce(Admittance):
    def __init__(self, M, D, K, h, method=DiscretizationMethod.FORWARD) -> None:
        super().__init__(M, D, K, h, 3, method)
    
    def compute(self, f: np.ndarray, h_new: float = None):    
        """ Returns position and velocity given an input force.
            This function can run at minimum 14kHz in "forward" mode on a decent laptop.
        """
        return super().compute(f, h_new)

class AdmittanceTorque(Admittance):
    
    def __init__(self, M, D, K, h, method=DiscretizationMethod.FORWARD) -> None:
        super().__init__(M, D, K, h, 3, method)
    
    def compute(self, m: np.ndarray, h_new: float = None):    
        """ Returns rotation and angular velocity given an input torque.
            This function can run at minimum 13kHz in "forward" mode on a decent laptop.
        """
        # q = log(Q), dq is its derivative
        # Q is the rotation quaternion, W (omega, the angular velocity) is its derivative
        q, dq = super().compute(m, h_new)
        Q = np.exp(quat.quaternion(0, *(0.5*q)))  # `quat.from_rotation_vector` is slow, do it manually
        # w = 2*quat.quaternion(*(quaternions.jac_q(q) @ dq)) * Q.conj()
        # W = w.vec
        W = dq
        return Q, W

    def compute_signal(self, U: np.ndarray):
        T = U.shape[0]
        X = np.zeros((T,), dtype=np.quaternion)
        dX = np.zeros((T, self.dims))
        for t in range(T):
            X[t, ...], dX[t, ...] = self.compute(U[t, ...])
        return X, dX

# TODO implement an `AdmittanceWrenchScrew`
class AdmittanceWrenchDecoupled(Admittance):
    def __init__(self, M, D, K, h, method=DiscretizationMethod.FORWARD) -> None:
        super().__init__(M, D, K, h, 6, method)
    
    def compute(self, w: np.ndarray, h_new: float = None):    
        """ Returns position and velocity given an input wrench.
            This function can run at minimum 6kHz in "forward" mode on a decent laptop.
        """
        x, dx = super().compute(w, h_new)
        p, dp, q, dq = x[:3], dx[:3], x[3:], dx[3:]
        Q = np.exp(quat.quaternion(0, *(0.5*q)))  # `quat.from_rotation_vector` is slow, do it manually
        # w = 2*quat.quaternion(*(quaternions.jac_q(q) @ dq)) * Q.conj()
        # W = w.vec
        W = dq
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