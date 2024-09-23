import numpy as np

from integrators import EUL, RK45


def estimate_derivatives(x, t):
    # time step
    t_delta = t[1:] - t[:-1]
    t_delta = np.concatenate([t_delta[[0]], t_delta])[:, np.newaxis]
    # velocity
    v0 = -1.5*x[[0], :] + 2*x[[1], :] - 0.5*x[[2], :]
    vt = -0.5*x[:-2, :] + 0.5*x[2:, :]
    vT = 0.5*x[[-3], :] - 2*x[[-2], :] + 1.5*x[[-1], :]
    v = np.concatenate([v0, vt, vT]) / t_delta
    #
    a0 = 2*x[[0], :] - 5*x[[1], :] + 4*x[[2], :] - 1*x[[3], :]
    at = 1*x[:-2, :] - 2*x[1:-1, :] + 1*x[2:, :]
    aT = -1*x[[-4], :] + 4*x[[-3], :] - 5*x[[-2], :] + 2*x[[-1], :]
    a = np.concatenate([a0, at, aT]) / t_delta**2
    return x, v, a


# noinspection PyPep8Naming
class DMP(object):

    def __init__(self, n_dim, K, n_basis, alpha=4):
        """
        Initialize the DMP object. The user should be able to set
        * the dimension of the system,
        * elastic term (damping term is automatically set to have critical damping)
        * number (and, optionally, parameters) of basis functions
        * decay parameter of the canonical system
        """
        self.n_dim = n_dim
        self.elastic_constant = K
        self.damping_constant = 2 * np.sqrt(self.elastic_constant)
        self.n_basis = n_basis
        self.alpha_cs = alpha
        self.T = 1

        # default parameters
        self.learned_x0 = None
        self.learned_xgoal = None
        self.weights = np.zeros([self.n_dim, self.n_basis+1])  # weights of the dynamic
        self.obstacles = lambda x, v: 0  # no obstacles

    def exp_basis(self, s):
        i = np.arange(0, self.n_basis + 1)
        c = np.exp(-self.alpha_cs * i * self.T / self.n_basis)
        h = np.zeros(self.n_basis + 1)
        h[:-1] = 1 / (c[1:] - c[:-1]) ** 2
        h[-1] = h[-2]
        return np.exp(-h * (s - c) ** 2)

    def compute_basis_vector(self, s):
        # TODO allow for custom basis functions
        basis = self.exp_basis(s)
        Phi = basis / np.sum(basis, axis=1, keepdims=True) * s
        return Phi

    def compute_perturbation(self, s):
        # weights = [dim]x[N+1]
        # basis   = [N+1]x[1]
        Phi = self.compute_basis_vector(s)

        #   [dim]x[1] / [1] * [1]
        f = self.weights @ Phi.T
        if f.shape[1] == 1:
            f = f[:, 0]
        return f

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles

    def learn_trajectory(self, desired_behavior, time_span):
        """
        This method should compute the set of weights given a desired behavior.

        desired_behavior is a [time]*[dim] matrix
        """
        #############################################
        # Step 1 : extract the desired forcing term #
        #############################################

        # prep
        x0 = desired_behavior[0, :]
        g = desired_behavior[-1, :]
        self.learned_x0 = x0
        self.learned_xgoal = lambda t: g

        K = self.elastic_constant * np.eye(self.n_dim)
        D = self.damping_constant * np.eye(self.n_dim)
        self.T = time_span[-1]

        # basically, we use this regressor
        # f(s) = W*Phi(s)
        # Phi(s) = [ phi_i(s) / SUM_i(phi_i(s)) ]

        s0 = 1
        tau = 1
        s = s0 * np.exp(-self.alpha_cs / tau * time_span)[:, np.newaxis]

        # compute desired perturbations
        x, v, a = estimate_derivatives(desired_behavior[:, 0:self.n_dim], time_span)
        # x = desired_behavior[:, 0:2]
        # v = np.array([
        #     time_span,                              # x
        #     2*np.sin(time_span)*np.cos(time_span)   # y
        # ]).T
        # a = np.array([
        #     time_span,                                      # x
        #     2*np.cos(time_span)**2-2*np.sin(time_span)**2   # y
        # ]).T
        f = (a + v @ D) @ np.linalg.inv(K) - (g - x) + (g - x0) * s

        # compute the basis vector
        Phi = self.compute_basis_vector(s)

        ##############################################################################
        # Step 2 : compute the set of weights starting from the desired forcing term #
        ##############################################################################
        F, P = f.T, Phi.T
        P_pinv = P.T @ np.linalg.inv(P @ P.T)
        weights = F @ P_pinv
        return weights

    def execute_trajectory(self, x0, xgoal, t_delta=None, tau=1, tol=1e-3, use_euler=False):
        """
        This method should return an execution of the dynamical system.
        The system should evolve until convergence (within a given tolerance) to the goal is achieved.
        """

        def dynamics(t, z, g, K, D):
            v = z[0:self.n_dim]
            x = z[self.n_dim:2 * self.n_dim]
            s = z[-1:]
            f = self.compute_perturbation(s[:, np.newaxis])
            dyn = np.concatenate([
                K @ (g(t) - x) - D @ v - K @ (g(t) - x0) * s + K @ f + self.obstacles(x, v),
                v,
                -self.alpha_cs * s
            ]) / tau
            return dyn

        return self._execute(dynamics, x0, xgoal, t_delta, tol, use_euler)

    def execute_trajectory_scaled(self, x0, xgoal, t_delta=None, tau=1, tol=1e-3, use_euler=False):
        """
        This method should return an execution of the dynamical system.
        The system should evolve until convergence (within a given tolerance) to the goal is achieved.
        """

        scalability = self.get_scalability_function()

        def dynamics(t, z, g, K, D):
            v = z[0:self.n_dim]
            x = z[self.n_dim:2 * self.n_dim]
            s = z[-1:]

            # change the system
            S = scalability(self.learned_x0, self.learned_xgoal(t), x0, xgoal(t))
            S_inv = np.linalg.inv(S)
            K = S @ K @ S_inv
            D = S @ D @ S_inv
            f = S @ self.compute_perturbation(s[:, np.newaxis])

            dyn = np.concatenate([
                K @ (g(t) - x) - D @ v - K @ (g(t) - x0) * s + K @ f + self.obstacles(x, v),
                v,
                -self.alpha_cs * s
            ]) / tau
            return dyn

        return self._execute(dynamics, x0, xgoal, t_delta, tol, use_euler)

    def _execute(self, dynamics, x0, xgoal, t_delta=None, tol=1e-3, use_euler=False):
        """
        This method should return an execution of the dynamical system.
        The system should evolve until convergence (within a given tolerance) to the goal is achieved.
        """

        D = self.damping_constant * np.eye(self.n_dim)
        K = self.elastic_constant * np.eye(self.n_dim)

        v0 = np.zeros(shape=(self.n_dim,))
        s0 = np.ones(shape=(1,))
        t0 = 0

        z0 = np.concatenate([v0, x0, s0])
        dyn = lambda t, z: dynamics(t, z, xgoal, K, D)  # time is not important here
        cond = lambda t, z: np.linalg.norm(z[self.n_dim:2 * self.n_dim] - xgoal(t)) <= tol  # ||x - g|| <= tol

        if use_euler:
            solver = EUL(h_init=t_delta)
        else:
            solver = RK45(h_init=t_delta)
        time_span, z_span = solver(dyn, z0, t0, final_cond=cond)
        return time_span, z_span

    def get_scalability_function(self):
        if self.n_dim == 2:
            return self.scalability2d
        elif self.n_dim == 3:
            return self.scalability3d
        else:
            raise Exception("Only 2D and 3D scalability is implemented")

    def scalability2d(self, x0, g, x0_new, g_new):

        # versors
        v = g - x0
        v_new = g_new - x0_new

        # 2D rotation matrix around the Z axis
        th = np.arccos(v @ v_new / (np.linalg.norm(v) * np.linalg.norm(v_new)))

        R = np.array([
            [np.cos(th), -np.sin(th)],
            [np.sin(th), np.cos(th)]
        ])
        l = np.linalg.norm(v_new) / np.linalg.norm(v)
        S = l * R
        return S

    def scalability3d(self, x0, g, x0_new, g_new):

        # versors
        a = (g - x0) / np.linalg.norm(g - x0)
        b = (g_new - x0_new) / np.linalg.norm(g_new - x0_new)

        # 3D rotation matrix that sends  a  to  b  around the axis between them
        c = np.dot(a, b)
        s = np.linalg.norm(np.cross(a, b))
        R = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
        # transform the rotation on the canonical axis
        u = a
        v = (b - (a @ b) * a) / np.linalg.norm(b - (a @ b) * a)
        w = np.cross(u, v)
        F = np.zeros((3, 3))
        F[:, 0] = u
        F[:, 1] = v
        F[:, 2] = w
        R = np.linalg.inv(F) @ R @ F

        # create the final matrix
        l = np.linalg.norm(g - x0) / np.linalg.norm(g_new - x0_new)
        S = l * R
        return S
