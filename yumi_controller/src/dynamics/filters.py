import numpy as np
from . import quat_utils


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
    
    def compute(self, u, h=None) -> float:
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


class DiscretizedStateSpaceModel(object):
    """ Implementation of a discretized time-invariant state-space model.
        Discretization can be either forward/backward Euler or Tustin.
        More at https://en.wikipedia.org/wiki/Discretization
    """
    
    def __init__(self, 
        A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, h: float, 
        x0: np.ndarray = None, 
        method="tustin"
    ) -> None:
        """ Create a n-dof dimensional state-space model
            :param A: state commutation matrix
            :param B: input commutation matrix
            :param C: state-output matrix (can be None)
            :param D: input-output matrix (can be None)
            :param h: step size
            :param method: approximation method {exact, forward, backward, tustin}
        """
        ALLOWED_METHODS = ["exact", "forward", "backward", "tustin"]
        
        assert C is None == D is None, "Both C and D must be None"
        
        assert A.ndim == 2 and B.ndim == 2, "All matrices must be 2-dimensional"
        assert A.shape[0] == A.shape[1], "A must be a square matrix"
        assert A.shape[0] == B.shape[0], "B matrix must have same rows as A"
        if C is not None:  # and D is not None
            assert C.ndim == 2 and D.ndim == 2, "All matrices must be 2-dimensional"
            assert A.shape[0] == C.shape[1], "C matrix must have columns as A matrix rows"
            assert C.shape[0] == D.shape[0], "D matrix must have same rows as C"
            assert B.shape[1] == D.shape[1], "D matrix must have same columns as B"
        assert method in ALLOWED_METHODS, f"method must be one of {{ {', '.join(ALLOWED_METHODS)} }}"
        
        self.A, self.B, self.C, self.D = A, B, C, D
        self.n, self.m, self.p = A.shape[0], B.shape[1], C.shape[0] if C is not None else 0
        self.method = method
        self._setup_coeffs(h)
        
        # previous state
        self.x0 = x0 if x0 is not None else np.zeros((self.n,1))
        self.x: np.ndarray
        self.clear()
        
    def _setup_coeffs(self, h: float) -> None:
        self.h = h
        # actual coefficient used for computation
        if self.method == "exact":
            # TODO here we SUPPOSE  A  IS DIAGONALIZABLE
            L, V = np.linalg.eig(self.A)  # A = V @ diag(E)*h @ inv(V)
            eAh = V @ np.exp(np.diag(L*h)) @ np.linalg.inv(V)
            F = eAh
            G = np.linalg.inv(self.A) @ (eAh - np.eye(self.n)) @ self.B
            # FIXME wtf happened here?
            raise RuntimeError("exact method is currently broken")
        elif self.method == "forward":
            eAh = np.eye(self.n) + self.A * h
            F = eAh
            G = self.B * h
        elif self.method == "backward":
            eAh = np.linalg.inv(np.eye(self.n) - self.A * h)
            F = eAh
            G = eAh @ self.B * h  
        elif self.method == "tustin":
            eAh = (np.eye(self.n) + 0.5 * self.A * h) @ np.linalg.inv(np.eye(self.n) - 0.5 * self.A * h)
            F = eAh
            G = np.linalg.inv(self.A) @ (eAh - np.eye(self.n)) @ self.B
        else:
            raise RuntimeError("no such method found")
        
        self.F = F
        self.G = G
    
    def clear(self):
        self.x = self.x0
    
    def __call__(self, u: np.ndarray, h_new: float = None, return_output: bool = True):
        """ Returns the state (and the output) of the system for a given input.
        """
        if h_new is not None:
            self._setup_coeffs(h_new)
        if u.ndim == 1:
            u = np.expand_dims(u, axis=-1)
        # compute the new state
        self.x = self.F @ self.x + self.G @ u
        # compute the new output
        if return_output and self.C is not None:
            y = self.C @ self.x + self.D @ u
            return np.squeeze(self.x, axis=-1), y
        else:
            return np.squeeze(self.x, axis=-1)
    
    def compute(self, u: np.ndarray, h_new: float = None):
        """ Returns the state of the system for a given input.
            Override this method if output from __call__ must be manipulated.
        """
        return self(u, h_new, False)

    def compute_signal(self, U: np.ndarray):
        """ Compute the system over the provided time-series
            :param U: time-series with shape(?,self.m)
        """
        X = []
        for t in range(U.shape[0]):
            X.append(self.compute(U[t, ...]))
        return X


class Admittance(DiscretizedStateSpaceModel):
    
    def __init__(self, m, k, d, h, n=None, method="tustin") -> None:
        """ Create a n-dof dimensional admittance
            :param m: mass of the admittance (float, 1-d, or 2-d ndarray)
            :param k: spring of the admittance (float, 1-d, or 2-d ndarray)
            :param n: size of the input (int or None)
            :param d: damping of the admittance (float, 1-d, 2-d ndarray, or None for critically damped system)
            :param h: step size (float or 1-d ndarray)
            :param method: approximation method {exact, forward, backward, tustin}
        """
        def reshape(x, n):
            x: np.ndarray = np.asarray(x)
            if x.ndim == 0:
                x = x * np.eye(n)
            elif x.ndim == 1 and x.shape == (n,):
                x = np.diag(x)
            elif x.ndim == 2 and x.shape == (n,n):
                pass
            else:
                raise ValueError(f"shape {x.shape} is not consistent with size n={n}")
            return x

        def matrix_sqrt(x):
            # Computing diagonalization
            E, V = np.linalg.eig(x)
            # Ensuring square root matrix exists
            assert np.all(E >= 0)
            sqrt_matrix = V * np.sqrt(E) @ np.linalg.inv(V)
            return sqrt_matrix
        
        if n is None:
            # the first tuple is to ensure at least one dimension
            n = np.max(np.concatenate([(1,), np.shape(m), np.shape(k), () if d is None else np.shape(d)])).astype(int)
        
        self.m = reshape(m, n)
        self.k = reshape(k, n)
        self.d = 2*matrix_sqrt((self.m @ self.k)) if d is None else reshape(d, n)
        self.ndim = n
        
        invM = np.linalg.inv(self.m)
        A = np.block([[np.zeros((n,n)), np.eye(n)], [-invM @ self.k, -invM @ self.d]])
        B = np.block([[np.zeros((n,n))], [invM]])
        
        super().__init__(A, B, None, None, h, None, method)
    
    def compute(self, u: np.ndarray, h_new: float = None):  
        """ Returns "position" and "velocity" given an input.
        """
        x = super().compute(u, h_new)
        y, dy = x[:self.ndim,...], x[self.ndim:,...]
        return y, dy
    
    def compute_signal(self, U: np.ndarray):
        X = super().compute_signal(U)
        Y, DY = map(list, zip(*X))
        return np.array(Y), np.array(DY)


class AdmittanceForce(Admittance):
    def __init__(self, m, k, d, h, method="tustin") -> None:
        super().__init__(m, k, d, h, 3, method)
    
    def compute(self, f: np.ndarray, h_new: float = None):    
        """ Returns position and velocity given an input force.
        """
        return super().compute(f, h_new)


class AdmittanceTorque(Admittance):
    
    def __init__(self, m, k, d, h, method="tustin") -> None:
        super().__init__(m, k, d, h, 3, method)
    
    def compute(self, m: np.ndarray, h_new: float = None):    
        """ Returns rotation and angular velocity given an input torque.
        """
        # TODO switch to np.quaternion here
        # q = log(Q), dq is its derivative
        # Q is the rotation quaternion, W (omega, the angular velocity) is its derivative
        q, dq = super().compute(m, h_new)
        Q = quat_utils.exp(q)
        W = 2*quat_utils.mult((quat_utils.jac_q(q) @ dq), quat_utils.conj(Q))
        w = W[1:]
        return Q, w


class AdmittanceWrench(Admittance):
    def __init__(self, m, k, d, h, method="tustin") -> None:
        super().__init__(m, k, d, h, 6, method)
    
    def compute(self, f: np.ndarray, h_new: float = None):    
        """ Returns position and velocity given an input force.
        """
        q, dq = super().compute(f, h_new)
        Q = quat_utils.exp(q[3:])
        W = 2*quat_utils.mult((quat_utils.jac_q(q[3:]) @ dq[3:]), quat_utils.conj(Q))
        w = W[1:]
        return (q[:3], Q), (dq[:3], w)


def _main_admittance_tustin():
    # Just a simple test for the admittance. 
    # Here we create a 3d admittance 
    import numpy as np
    import matplotlib.pyplot as plt
    
    m = 2  # we can use floats ...
    k = np.array([1000, 5000, 700])  # ... or Numpy array (everything will be broadcasted to the right shape)
    k = 1000
    d = 2*np.sqrt(m*k)  # for a critically damped admittance
    h = 1/1000  # sampling step [s]
    
    adm = AdmittanceTustin(m, k, d, h)  # d=None yields the same result
    
    # three noisy box signals sampled with step h
    # each will pass through a different admittance
    t = np.linspace(0, 2.5, int(1/h), endpoint=True)
    u = np.zeros((t.shape[0], 3))
    u[t >= 0.5, ...] = [25, 15, 10]
    u[t >  1.5, ...] = 0
    u += np.random.normal(0, 0.025, size=u.shape)
    
    # calculate the output signal 
    y = np.zeros_like(u)
    dy = np.zeros_like(u)
    for i in range(u.shape[0]):
        y[i, ...], dy[i, ...], _ = adm.compute(u[i, ...])
    
    # plot the result
    fig, ax1 = plt.subplots()
    ax1.set_frame_on(True)
    ax1.plot(t, u)
    ax2 = ax1.twinx()
    ax2.spines["right"].set_position(("axes", 1.0))
    ax2.plot(t, y, linestyle="--")
    ax2.plot(t, dy, linestyle="-.")
    plt.show()

def _main_lpfilter():
    # Just a simple test for the admittance. 
    # Here we create a 3d admittance 
    import numpy as np
    import matplotlib.pyplot as plt
    
    f = 5
    k = 1
    h = 1/1000  # sampling step [s]
    
    lp = LPFilterTustin(f, k, h)
    
    # three noisy box signals sampled with step h
    # each will pass through a different admittance
    t = np.linspace(0, 5, int(1/h), endpoint=True)
    u = np.sin(2*np.pi*15 * t)
    
    # calculate the output signal 
    y = np.zeros_like(u)
    for i in range(u.shape[0]):
        y[i, ...] = lp(u[i, ...])
    
    # plot the result
    fig, ax1 = plt.subplots()
    ax1.set_frame_on(True)
    ax1.plot(t, u)
    ax2 = ax1.twinx()
    ax2.spines["right"].set_position(("axes", 1.0))
    ax2.plot(t, y, linestyle="dashed")
    plt.show()

def _main_admittance_force():
    # Just a simple test for the admittance. 
    # Here we create a 3d admittance 
    import numpy as np
    import matplotlib.pyplot as plt
    
    m = 2
    k = 500 #np.diag([800, 1000, 1200])
    d = None
    h = 1/1000  # sampling step [s]
    adm = AdmittanceForce(m, k, d, h, method="tustin")
    
    # three noisy box signals sampled with step h
    # each will pass through a different admittance
    t = np.linspace(0, 2.5, int(1/h), endpoint=True)
    u = np.zeros((t.shape[0], 3))
    u[t >= 0.5, ...] = [25, 0, 0]
    u[t >  1.5, ...] = 0
    u += np.random.normal(0, 0.025, size=u.shape)
    
    # calculate the output signal
    y, dy = adm.compute_signal(u)
    
    # plot the result
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(t, u, label=r"$u$")
    ax[0].legend()
    ax[1].plot(t, y, linestyle="-", label=r"$y$")
    ax[1].plot(t, dy, linestyle="--", label=r"$\dot{y}$")
    ax[1].legend()
    plt.show()

def _main_admittance_torque():
    # Just a simple test for the admittance. 
    # Here we create a 3d admittance 
    import numpy as np
    import matplotlib.pyplot as plt
    
    m = 0.01
    k = 100 #np.diag([800, 1000, 1200])
    d = None
    h = 1/1000  # sampling step [s]
    
    adm = AdmittanceTorque(m, k, d, h, method="tustin")
    
    # three noisy box signals sampled with step h
    # each will pass through a different admittance
    t = np.linspace(0, 2.5, int(1/h), endpoint=True)
    u = np.zeros((t.shape[0], 3))
    u[t >= 0.5, ...] = [10, 10, 0]
    u[t >  1.5, ...] = 0
    u += np.random.normal(0, 0.025, size=u.shape)
    
    # calculate the output signal
    Q, w = adm.compute_signal(u)
    # plot the result
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(t, u, label=r"$u$")
    ax[0].legend()
    ax[1].plot(t, Q, linestyle="-", label=r"$y$")
    #ax[1].plot(t, w, linestyle="--", label=r"$\dot{y}$")
    ax[1].legend()
    plt.show()
    


if __name__ == "__main__":
    _main_admittance_torque()