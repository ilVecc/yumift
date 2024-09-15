###
# TODO expand this and it a package 
###
import numpy as np

class Integrator(object):
      
    def step(self, f, y0, t0, h):
        raise NotImplementedError("Integration method not implemented")
    
    def compute(self, f, y0, t0=0, tf=float("inf"), final_cond=None):
        h = self.h_init
        
        tt = [t0]
        yy = np.zeros(shape=(1, y0.shape[0]))
        yy[0, :] = y0

        while True:
            # get the current state
            t = tt[-1]
            y = yy[-1, :]
            
            # compute a step
            t_now, y_now, h_now = self.step(f, y, t, h)
            
            # save the next state
            h = h_now
            tt.append(t_now)
            yy = np.concatenate([yy, y_now[np.newaxis, :]])

            # check the termination conditions
            step_condition = h_now > 0
            time_condition = t_now < tf
            goal_condition = final_cond is None or not final_cond(t_now, y_now)
            if not time_condition or not goal_condition or not step_condition:
                break

        return np.array(tt), yy
    
    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)

class EUL(Integrator):
    def __init__(self, h_init):
        super(Integrator, self).__init__()
        self.h_init = h_init
    
    def step(self, f, y0, t0, h):
        t = t0
        y = y0
        
        # evolution using Euler
        y_now = y + h * f(t, y)
        t_now = t + h
        
        return t_now, y_now, h

# noinspection PyPep8Naming
class RK45(Integrator):
    
    def __init__(self, h_init=1e-3, h_min=1e-10, e_tol=1e-6):
        super(Integrator, self).__init__()
        self.h_init = h_init
        self.h_min = h_min
        self.e_tol = e_tol
        
        self.tableau_45_C = np.array([
            0, 1/4, 3/8, 12/13, 1, 1/2  # sub-steps coefficients for each ki
        ])
        self.tableau_45_A = np.array([
            [        0,          0,          0,         0,      0],  # k1
            [      1/4,          0,          0,         0,      0],  # k2
            [     3/32,       9/32,          0,         0,      0],  # k3
            [1932/2197, -7200/2197,  7296/2197,         0,      0],  # k4
            [  439/216,         -8,   3680/513, -845/4104,      0],  # k5
            [    -8/27,          2, -3544/2565, 1859/4104, -11/40]   # k6
        ])
        self.tableau_45_B = np.array([
            [  16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55],  # order 5
            [  25/216, 0,  1408/2565,   2197/4104,  -1/5,    0]   # order 4
        ])
    
    def step(self, f, y0, t0, h):
        
        s = self.tableau_45_C.shape[0]
        K = np.zeros(shape=(s, y0.shape[0]))
        
        t = t0
        y = y0
        
        # find a good time step and advance
        j = 0
        can_continue = True
        while True:
            # compute coefficients
            for i in range(s):
                k_t = t + h * self.tableau_45_C[i]
                k_y = y + h * self.tableau_45_A[i, :] @ K[:-1, :]
                K[i, :] = f(k_t, k_y)

            # compute next step
            y_now_higher = y + h * self.tableau_45_B[0, :] @ K
            y_now_lower = y + h * self.tableau_45_B[1, :] @ K

            # compute truncation error and new step size
            # if the error is lower than the tolerance, move to the next step
            # if not, recompute this step with the new step size (h_new)
            trunc_error = np.linalg.norm(y_now_higher - y_now_lower) / np.linalg.norm(y_now_higher)
            if trunc_error <= self.e_tol:
                h *= 1.1
                break
            else:
                h *= 0.9

            j += 1
            if j > 1000:
                print("Too many loops for h!")
                can_continue = False
                break

            if h < self.h_min:
                print("Reached too small h!")
                can_continue = False
                break

        # save the state
        t_now = t + h
        y_now = y_now_higher
        
        # step condition
        if not can_continue:
            h = -1
        
        return t_now, y_now, h
    