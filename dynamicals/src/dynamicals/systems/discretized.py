from typing import Tuple

import numpy as np


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