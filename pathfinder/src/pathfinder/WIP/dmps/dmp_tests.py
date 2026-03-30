import numpy as np
import matplotlib.pyplot as plt
from dmp import DMP, estimate_derivatives


# PLOTTINGS

def plot_evolutions(trjs, legend=None):
    plt.figure()
    for t, x, style in trjs:
        plt.plot(t, x, style)
    if legend is not None:
        plt.legend(legend)
    plt.xlabel("t")
    plt.show()


def plot_trajectories(trjs, legend=None, extra_plot_handles=None, show_endpoints=True):
    plt.figure()

    # choose between 2D or 3D
    n_dims = trjs[0][1].shape[1]
    if n_dims == 2:
        ax = plt.gca()
        plt.axis("equal")
    elif n_dims == 3:
        ax = plt.axes(projection="3d")
        ax.view_init(15, 30)
        plt.axis("auto")
    else:
        raise Exception("Cannot plot in more than 3 dimensions!")

    # for obstacles or anything else
    if extra_plot_handles is not None:
        extra_plot_handles()

    # for the trajectories
    if n_dims == 3:
        for t, x, style in trjs:
            if show_endpoints:
                ax.plot3D(x[0, 0], x[0, 1], x[0, 2], "ok")
                ax.plot3D(x[-1, 0], x[-1, 1], x[-1, 2], "xk")
            ax.plot3D(x[:, 0], x[:, 1], x[:, 2], style)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    else:
        for t, x, style in trjs:
            if show_endpoints:
                ax.plot(x[0, 0], x[0, 1], "ok")
                ax.plot(x[-1, 0], x[-1, 1], "xk")
            ax.plot(x[:, 0], x[:, 1], style)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    # show the legend if necessary
    if legend is not None:
        plt.legend(legend)
    plt.show()


def plot_obstacle_point(o, show_3D=False):
    if show_3D:
        plt.gca().plot3D(o[0], o[1], o[2], "or")
    else:
        plt.plot(o[0], o[1], "or")


def plot_moving_goal(time_span, goal):
    goal_span = np.array([goal(t) for t in time_span])
    if goal_span.shape[1] == 3:
        plt.gca().plot3D(goal_span[:, 0], goal_span[:, 1], goal_span[:, 2], ":k")
    else:
        plt.plot(goal_span[:, 0], goal_span[:, 1], ":k")


# TESTS

def test_default_trajectory():
    #################################################################
    # try the learning process on the un-perturbed scenario (W = 0) #
    # so to find out the errors in the estimation                   #
    #################################################################

    # create the DMP
    K = 150
    n_basis = 50
    alpha = 4

    n_dim = 2
    dmp = DMP(n_dim, K, n_basis, alpha)

    # look at the default trajectory
    x0 = np.zeros(shape=(n_dim,))
    xgoal = lambda t: np.ones(shape=(n_dim,))
    t_dmp, trj_dmp = dmp.execute_trajectory(x0, xgoal, t_delta=1e-3, tol=1e-5)

    # learn the weights using the default trajectory and re-compute the trajectory
    dmp.weights = dmp.learn_trajectory(trj_dmp[:, 2:4], t_dmp)
    t_dmp_new, trj_dmp_new = dmp.execute_trajectory(x0, xgoal, t_delta=1e-3, tol=1e-5)

    # compare the trajectories
    plot_trajectories([
        (t_dmp, trj_dmp[:, 2:4], "--b"),
        (t_dmp_new, trj_dmp_new[:, 2:4], "r")
    ])
    plot_evolutions([
        (t_dmp, trj_dmp[:, 2:4], "--b"),
        (t_dmp_new, trj_dmp_new[:, 2:4], "r")
    ],
        legend=["x_default", "y_default", "x_dmp", "y_dmp"]
    )


def test_simple_trajectory():
    # Model parameters
    K = 150
    n_basis = 50
    alpha = 4

    ##################
    # LEARNING PHASE #
    ##################

    # Desired behavior, shaped as a (T x n_dim) array
    time_span = np.linspace(0, np.pi, 100)
    trj_desired = np.array([
        time_span,              # x
        np.sin(time_span) ** 2  # y
    ]).T

    # Learning
    dmp = DMP(trj_desired.shape[1], K, n_basis, alpha)
    dmp.weights = dmp.learn_trajectory(trj_desired, time_span)

    ###################
    # EXECUTION PHASE #
    ###################

    # Execution
    x0 = np.array([0.0, 0.0])
    xgoal = lambda t: np.array([3.6, 0.2])
    t_dmp, trj_dmp = dmp.execute_trajectory(x0, xgoal, t_delta=1e-2, tol=1e-2)

    plot_trajectories([
        (time_span, trj_desired[:, 0:2], "--b"),
        (t_dmp, trj_dmp[:, 2:4], "r")
    ])
    plot_evolutions([
        (t_dmp, trj_dmp[:, 2:4], "r"),
        (t_dmp, trj_dmp[:, 0:2], "b"),
        (t_dmp, trj_dmp[:, 4], "g")
    ],
        legend=["x(t)", "y(t)", "vx(s)", "vy(s)", "s(t)"]
    )


def test_simple_trajectory_robustness(random_start=True, random_goal=True):
    # Model parameters
    K = 150
    n_basis = 50
    alpha = 4

    ##################
    # LEARNING PHASE #
    ##################

    # Desired behavior, shaped as a (T x n_dim) array
    time_span = np.linspace(0, np.pi, 100)
    trj_desired = np.array([
        time_span,              # x
        np.sin(time_span) ** 2  # y
    ]).T

    # Learning
    dmp = DMP(trj_desired.shape[1], K, n_basis, alpha)
    weights = dmp.learn_trajectory(trj_desired, time_span)
    dmp.weights = weights

    # Many executions
    trajectories = []
    for i in range(25):
        x0 = np.array([0.0, 0.0]) + (np.random.random(2) * 0.5 - 0.25) * random_start
        new_goal = np.array([np.pi, 0.0]) + (np.random.random(2) * 0.5 - 0.25) * random_goal
        xgoal = lambda t: new_goal
        t_dmp, trj_dmp = dmp.execute_trajectory(x0, xgoal, t_delta=1e-2, tol=1e-2)

        trajectories.append((t_dmp, trj_dmp[:, 2:4], ""))

    plot_trajectories(trajectories)


def test_simple_3D_trajectory():
    # Model parameters
    K = 150
    n_basis = 50
    alpha = 4

    ##################
    # LEARNING PHASE #
    ##################

    # Desired behavior, shaped as a (T x n_dim) array
    time_span = np.linspace(0, 2*np.pi, 200)
    trj_desired = np.array([
        np.cos(time_span),  # x
        np.sin(time_span),  # y
        time_span           # z
    ]).T

    # Learning
    dmp = DMP(trj_desired.shape[1], K, n_basis, alpha)
    dmp.weights = dmp.learn_trajectory(trj_desired, time_span)

    ###################
    # EXECUTION PHASE #
    ###################

    # Execution
    x0 = np.array([1.0, 0.0, 0.0])
    xgoal = lambda t: np.array([1.0, 0.2, 2*np.pi+1])
    t_dmp, trj_dmp = dmp.execute_trajectory(x0, xgoal, t_delta=1e-2, tol=1e-2)

    plot_trajectories([
        (time_span, trj_desired, "--b"),
        (t_dmp, trj_dmp[:, 3:6], "r")
    ])


def test_ugly_trajectory():
    # Model parameters
    K = 150
    n_basis = 50
    alpha = 4

    ##################
    # LEARNING PHASE #
    ##################

    # Desired behavior, shaped as a (T x n_dim) array
    time_span = np.linspace(0, np.pi, 100)
    trj_desired = np.array([
        time_span,              # x
        np.sin(time_span) ** 2  # y
    ]).T

    # Learning
    dmp = DMP(trj_desired.shape[1], K, n_basis, alpha)
    dmp.weights = dmp.learn_trajectory(trj_desired, time_span)

    ###################
    # EXECUTION PHASE #
    ###################

    # Execution
    x0 = np.array([0.0, 0.0])
    xgoal = lambda t: np.array([-6, 8])
    t_dmp, trj_dmp = dmp.execute_trajectory(x0, xgoal, t_delta=1e-2, tol=1e-2)

    plot_trajectories([
        (time_span, trj_desired[:, 0:2], "--b"),
        (t_dmp, trj_dmp[:, 2:4], "r")
    ])


def test_ugly_3D_trajectory():
    # Model parameters
    K = 150
    n_basis = 50
    alpha = 4

    ##################
    # LEARNING PHASE #
    ##################

    # Desired behavior, shaped as a (T x n_dim) array
    time_span = np.linspace(0, 2*np.pi, 200)
    trj_desired = np.array([
        np.cos(time_span),  # x
        np.sin(time_span),  # y
        time_span           # z
    ]).T

    # Learning
    dmp = DMP(trj_desired.shape[1], K, n_basis, alpha)
    dmp.weights = dmp.learn_trajectory(trj_desired, time_span)

    ###################
    # EXECUTION PHASE #
    ###################

    # Execution
    x0 = np.array([1.0, 0.0, 0.0])
    xgoal = lambda t: np.array([1.0, 0.0, 2.6])
    t_dmp, trj_dmp = dmp.execute_trajectory(x0, xgoal, t_delta=1e-2, tol=1e-2)

    plot_trajectories([
        (time_span, trj_desired, "--b"),
        (t_dmp, trj_dmp[:, 3:6], "r")
    ])


# TEST SCALABILITY

def test_simple_scalable_trajectory():
    # Model parameters
    K = 150
    n_basis = 50
    alpha = 4

    ##################
    # LEARNING PHASE #
    ##################

    # Desired behavior, shaped as a (T x n_dim) array
    time_span = np.linspace(0, np.pi, 100)
    trj_desired = np.array([
        time_span,              # x
        np.sin(time_span) ** 2  # y
    ]).T

    # Learning
    dmp = DMP(trj_desired.shape[1], K, n_basis, alpha)
    dmp.weights = dmp.learn_trajectory(trj_desired, time_span)

    ###################
    # EXECUTION PHASE #
    ###################

    # Execution and application of affine transformation
    x0 = np.array([0.0, 0.0])
    xgoal = lambda t: np.array([-6, 8])
    t_dmp, trj_dmp = dmp.execute_trajectory(x0, xgoal, t_delta=1e-2, tol=1e-3)
    t_dmp_scaled, trj_dmp_scaled = dmp.execute_trajectory_scaled(x0, xgoal, t_delta=1e-2, tol=1e-3)

    # plotting
    plot_trajectories([
        (time_span, trj_desired, "--b"),
        (t_dmp, trj_dmp[:, 2:4], "r"),
        (t_dmp_scaled, trj_dmp_scaled[:, 2:4], "g")
    ])


def test_simple_3D_scalable_trajectory():
    # Model parameters
    K = 150.0
    n_basis = 50
    alpha = 4

    ##################
    # LEARNING PHASE #
    ##################

    # Desired behavior, shaped as a (T x n_dim) array
    time_span = np.linspace(0, 2*np.pi, 200)
    trj_desired = np.array([
        np.cos(time_span),  # x
        np.sin(time_span),  # y
        time_span           # z
    ]).T

    # Learning
    dmp = DMP(trj_desired.shape[1], K, n_basis, alpha)
    dmp.weights = dmp.learn_trajectory(trj_desired, time_span)

    ###################
    # EXECUTION PHASE #
    ###################

    # Execution
    x0 = np.array([1.0, 0.0, 0.0])
    xgoal = lambda t: np.array([1.0, 0.0, -2*np.pi])
    t_dmp, trj_dmp = dmp.execute_trajectory(x0, xgoal, t_delta=1e-2, tol=1e-2)
    t_dmp_scaled, trj_dmp_scaled = dmp.execute_trajectory_scaled(x0, xgoal, t_delta=1e-2, tol=1e-2)

    # plotting
    plot_trajectories([
        (time_span, trj_desired, "--b"),
        (t_dmp, trj_dmp[:, 3:6], "r"),
        (t_dmp_scaled, trj_dmp_scaled[:, 3:6], ":g")
    ])


# TEST MOVING GOAL

def test_scalability_moving_goal():
    # Model parameters
    K = 150
    n_basis = 50
    alpha = 4

    ##################
    # LEARNING PHASE #
    ##################

    # Desired behavior, shaped as a (T x n_dim) array
    time_span = np.linspace(0, np.pi, 100)
    trj_desired = np.array([
        time_span,              # x
        np.sin(time_span) ** 2  # y
    ]).T

    # Learning
    dmp = DMP(trj_desired.shape[1], K, n_basis, alpha)
    dmp.weights = dmp.learn_trajectory(trj_desired, time_span)

    ###################
    # EXECUTION PHASE #
    ###################

    # Execution

    def goal(t):
        t = min(1, t / np.pi)
        return 2*np.pi*np.array([np.cos(t), np.sin(t)])

    x0 = np.array([0.0, 0.0])
    xgoal = goal
    t_dmp, trj_dmp = dmp.execute_trajectory(x0, xgoal, t_delta=1e-2, tol=0.05)
    t_dmp, trj_dmp_scaled = dmp.execute_trajectory_scaled(x0, xgoal, t_delta=1e-2, tol=0.05)

    def extra_handles():
        plot_moving_goal(t_dmp, xgoal)

    plot_trajectories([
        (time_span, trj_desired[:, 0:2], "--b"),
        (t_dmp, trj_dmp[:, 2:4], "r"),
        (t_dmp, trj_dmp_scaled[:, 2:4], "g")
    ],
        extra_plot_handles=extra_handles
    )


def test_simple_3D_scalable_trajectory_moving_goal():
    # Model parameters
    K = 250
    n_basis = 50
    alpha = 4

    ##################
    # LEARNING PHASE #
    ##################

    # Desired behavior, shaped as a (T x n_dim) array
    time_span = np.linspace(0, 2*np.pi, 200)
    trj_desired = np.array([
        np.cos(time_span),  # x
        np.sin(time_span),  # y
        time_span           # z
    ]).T

    # Learning
    dmp = DMP(trj_desired.shape[1], K, n_basis, alpha)
    dmp.weights = dmp.learn_trajectory(trj_desired, time_span)

    ###################
    # EXECUTION PHASE #
    ###################

    def goal(t):
        ti = 0*np.pi
        tf = 3*np.pi
        return np.array([1.0, 0.0, -2*np.pi]) - 3.0*min((t - ti)/(tf - ti), 1) * np.array([0, 0, 1])

    x0 = trj_desired[0, :]
    xgoal = goal
    t_dmp, trj_dmp = dmp.execute_trajectory(x0, xgoal, t_delta=1e-2, tol=0.01)
    t_dmp_scaled, trj_dmp_scaled = dmp.execute_trajectory_scaled(x0, xgoal, t_delta=1e-2, tol=0.01)

    goal_span = np.array([goal(t) for t in t_dmp])

    def extra_handles():
        plt.gca().plot3D(goal_span[:, 0], goal_span[:, 1], goal_span[:, 2], ":k")

    # plotting
    plot_trajectories([
        (time_span, trj_desired, "--b"),
        (t_dmp, trj_dmp[:, 3:6], "r"),
        (t_dmp_scaled, trj_dmp_scaled[:, 3:6], ":g")
    ],
        extra_plot_handles=extra_handles
    )


# TEST OBSTACLES

def obstacle_point_static(o, beta):
    # The point static potential is formulated as
    #   Us(x) = eta / 2 * (1/rx - 1/r0) ** 2 if rx <= r0 else 0
    # with distance function of a circle
    #   r(x) = ||x - o||
    # We then must compute the force using
    #   p(x,v) = -∇x Us(x)
    r = lambda x: np.linalg.norm(x - o)
    p = lambda x, v: beta / r(x)**(beta+2) * (x - o)
    U = lambda x, v: beta / r(x)**beta
    return p, U


def obstacle_point_radial_static(o, eta, r0):
    # The point static potential is formulated as
    #   Us(x) = eta / 2 * (1/rx - 1/r0) ** 2 if rx <= r0 else 0
    # with distance function of a circle
    #   r(x) = ||x - o||
    # We then must compute the force using
    #   p(x,v) = -∇x Us(x)
    r = lambda x: np.linalg.norm(x - o)
    p = lambda x, v: eta * (1 / r(x) - 1 / r0) / r(x)**3 * (x - o) * (r(x) <= r0)
    U = lambda x, v: eta/2 * (1 / r(x) - 1 / r0) ** 2 * (r(x) <= r0)
    return p, U


def obstacle_point_dynamic(o, lambd, beta):
    # The point static potential is formulated as
    #   Ud(x) = lambda * (-cos(th)) ** beta * ||v||/r(x) if th in (pi/2, pi] else 0
    # with distance function of a circle
    #   r(x) = ||x - o||
    # and
    #   cos(th) = (v°x) / (||v||*r(x))
    # We then must compute the force using
    #   p(x,v) = -∇x Us(x)
    r = lambda x: np.linalg.norm(x - o)
    grad_r = lambda x: (x - o) / r(x)
    v_norm = lambda v: np.linalg.norm(v) if np.linalg.norm(v) != 0 else 1
    c = lambda x, v: np.dot(v, x) / (v_norm(v) * np.linalg.norm(x))

    # TODO this does not work! :(
    def potential(x, v):
        p = lambd * (-c(x, v)) ** (beta - 1) * v_norm(v) / r(x) ** 2 \
                * (beta * v / v_norm(v) - (beta + 1) * c(x, v) * grad_r(x)) * (-1 < c(x, v) <= 0)
        # print(p)
        return p

    p = lambda x, v: potential(x, v)
    U = lambda x, v: lambd * (-c(x, v)) ** beta * v_norm(v) / r(x) * (-1 < c(x, v) <= 0)
    return p, U


def test_ugly_obstacles():
    # Model parameters
    K = 150
    n_basis = 50
    alpha = 4

    # Desired behavior, shaped as a (T x n_dim) array
    radius = 2
    time_span = np.linspace(0, np.pi, 100)
    trj_desired = np.array([
        -radius * (1 - 2*np.cos(time_span) + np.cos(2*time_span)),  # x
        radius * (2*np.sin(time_span) - np.sin(2*time_span))        # y
    ]).T

    # Learning
    dmp = DMP(trj_desired.shape[1], K, n_basis, alpha)
    dmp.weights = dmp.learn_trajectory(trj_desired, time_span)

    # Add obstacle
    obs_strength = 50
    obs_radius = 0.5
    obs_c1 = trj_desired[75, :] + np.array([-0.5, 0.5])
    obs_p1 = obstacle_point_radial_static(obs_c1, eta=obs_strength, r0=obs_radius)[0]
    obs_c2 = trj_desired[40, :]
    obs_p2 = obstacle_point_radial_static(obs_c2, eta=obs_strength, r0=obs_radius)[0]
    obs_p = lambda x, v: obs_p1(x, v) + obs_p2(x, v)
    dmp.set_obstacles(obs_p)

    # Execution
    x0 = trj_desired[0, :]
    xgoal = lambda t: trj_desired[-1, :]
    t_dmp, trj_dmp = dmp.execute_trajectory(x0, xgoal, t_delta=1e-1, tol=1e-2, use_euler=True)

    def obs_plot():
        plot_obstacle_point(obs_c1)
        plot_obstacle_point(obs_c2)

    plot_trajectories([
        (time_span, trj_desired, "--b"),
        (t_dmp, trj_dmp[:, 2:4], "r")
    ],
        extra_plot_handles=obs_plot
    )


def test_obstacles():
    # Model parameters
    K = 150
    n_basis = 50
    alpha = 4

    # Desired behavior, shaped as a (T x n_dim) array
    radius = 2
    time_span = np.linspace(0, np.pi, 100)
    trj_desired = np.array([
        -radius * (1 - 2*np.cos(time_span) + np.cos(2*time_span)),  # x
        radius * (2*np.sin(time_span) - np.sin(2*time_span))        # y
    ]).T

    # Learning
    dmp = DMP(trj_desired.shape[1], K, n_basis, alpha)
    dmp.weights = dmp.learn_trajectory(trj_desired, time_span)

    # Add obstacle
    obs_strength = 750
    obs_radius = 1.5
    obs_c1 = trj_desired[75, :] + np.array([-0.5, 0.5])
    obs_p1 = obstacle_point_radial_static(obs_c1, eta=obs_strength, r0=obs_radius)[0]
    obs_c2 = trj_desired[40, :]
    obs_p2 = obstacle_point_radial_static(obs_c2, eta=obs_strength, r0=obs_radius)[0]
    obs_p = lambda x, v: obs_p1(x, v) + obs_p2(x, v)
    dmp.set_obstacles(obs_p)

    # Execution
    x0 = trj_desired[0, :]
    xgoal = lambda t: trj_desired[-1, :]
    t_dmp, trj_dmp = dmp.execute_trajectory(x0, xgoal, t_delta=1e-2, tol=1e-2)

    def obs_plot():
        plot_obstacle_point(obs_c1)
        plot_obstacle_point(obs_c2)

    plot_trajectories([
        (time_span, trj_desired, "--b"),
        (t_dmp, trj_dmp[:, 2:4], "r")
    ],
        extra_plot_handles=obs_plot
    )

#
# def test_obstacles_3D():
#     # Model parameters
#     K = 250.0
#     n_basis = 50
#     alpha = 4
#
#     # Desired behavior, shaped as a (T x n_dim) array
#     time_span = np.linspace(0, 2*np.pi, 200)
#     trj_desired = np.array([
#         np.cos(time_span),  # x
#         np.sin(time_span),  # y
#         time_span           # z
#     ]).T
#
#     # Learning
#     dmp = DMP(trj_desired.shape[1], K, n_basis, alpha)
#     dmp.weights = dmp.learn_trajectory(trj_desired, time_span)
#
#     # Add obstacles
#     obs_c1 = trj_desired[75, :]
#     obs_p1 = obstacle_point_static(obs_c1, beta=4)[0]
#     obs_c2 = trj_desired[40, :]
#     obs_p2 = obstacle_point_static(obs_c2, beta=4)[0]
#     obs_p = lambda x, v: obs_p1(x, v) + obs_p2(x, v)
#     dmp.set_obstacles(obs_p)
#
#     # Execution
#     def goal_1(t):
#         if t < np.pi:
#             return trj_desired[-1, :]
#         elif np.pi <= t < 1.5*np.pi:
#             return 1.2*trj_desired[-1, :]
#         else:
#             return trj_desired[-1, :]
#
#     def goal_2(t):
#         t1 = 1.5*np.pi
#         tf = 2*np.pi
#         if t < t1:
#             return trj_desired[-1, :]
#         else:
#             return trj_desired[-1, :] + 0.75*(t - t1)/(tf - t1) * np.array([0, 0, 1])
#
#     goal = goal_2
#
#     x0 = trj_desired[0, :]
#     xgoal = goal
#     t_dmp, trj_dmp = dmp.execute_trajectory(x0, xgoal, t_delta=1e-2, tol=1e-1)
#     goal_span = np.array([goal(t) for t in t_dmp])
#
#     def extra_handles():
#         plot_obstacle_point(obs_c1, show_3D=True)
#         plot_obstacle_point(obs_c2, show_3D=True)
#         plt.gca().plot3D(goal_span[:, 0], goal_span[:, 1], goal_span[:, 2], ":k")
#
#     plot_trajectories([
#         (time_span, trj_desired, "--b"),
#         (t_dmp, trj_dmp[:, 3:6], "r")
#     ],
#         extra_plot_handles=extra_handles
#     )
#

def test_obstacles_3D_moving_goal():
    # Model parameters
    K = 250.0
    n_basis = 50
    alpha = 4

    # Desired behavior, shaped as a (T x n_dim) array
    time_span = np.linspace(0, 2*np.pi, 200)
    trj_desired = np.array([
        np.cos(time_span),  # x
        np.sin(time_span),  # y
        time_span           # z
    ]).T

    # Learning
    dmp = DMP(trj_desired.shape[1], K, n_basis, alpha)
    dmp.weights = dmp.learn_trajectory(trj_desired, time_span)

    # Add obstacles
    obs_c1 = trj_desired[75, :]
    obs_p1 = obstacle_point_static(obs_c1, beta=4)[0]
    obs_c2 = trj_desired[40, :]
    obs_p2 = obstacle_point_static(obs_c2, beta=4)[0]
    # obs_p2 = obstacle_point_dynamic(obs_c2, lambd=25, beta=2)[0]
    obs_p = lambda x, v: obs_p1(x, v) + obs_p2(x, v)
    dmp.set_obstacles(obs_p)

    # Execution
    def goal_1(t):
        if t < np.pi:
            return trj_desired[-1, :]
        elif np.pi <= t < 1.5*np.pi:
            return 1.2*trj_desired[-1, :]
        else:
            return trj_desired[-1, :]

    def goal_2(t):
        t1 = 1.5*np.pi
        tf = 2*np.pi
        if t < t1:
            return trj_desired[-1, :]
        else:
            return trj_desired[-1, :] + 0.75*(t - t1)/(tf - t1) * np.array([0, 0, 1])

    goal = goal_2

    x0 = trj_desired[0, :]
    xgoal = goal
    t_dmp, trj_dmp = dmp.execute_trajectory(x0, xgoal, t_delta=1e-2, tol=1e-1)
    goal_span = np.array([goal(t) for t in t_dmp])

    def extra_handles():
        plot_obstacle_point(obs_c1, show_3D=True)
        plot_obstacle_point(obs_c2, show_3D=True)
        plt.gca().plot3D(goal_span[:, 0], goal_span[:, 1], goal_span[:, 2], ":k")

    plot_trajectories([
        (time_span, trj_desired, "--b"),
        (t_dmp, trj_dmp[:, 3:6], "r")
    ],
        extra_plot_handles=extra_handles
    )


# OTHER TESTS


def test_estimate_derivatives():
    time_span = np.linspace(0, np.pi, 100)
    trj_desired = np.array([
        time_span,  # x
        np.exp(time_span)  # y
    ]).T

    # real
    x_real = trj_desired[:, 0:2]
    v_real = np.array([
        time_span,          # x
        np.exp(time_span)   # y
    ]).T
    a_real = np.array([
        time_span,          # x
        np.exp(time_span)   # y
    ]).T

    # estimates
    x_est, v_est, a_est = estimate_derivatives(trj_desired[:, 0:2], time_span)

    # compute errors
    x_err = abs(x_real - x_est)
    v_err = abs(v_real - v_est)
    a_err = abs(a_real - a_est)

    plt.figure()
    plt.semilogy(time_span, x_err[:, 1], "r")
    plt.semilogy(time_span, v_err[:, 1], "b")
    plt.semilogy(time_span, a_err[:, 1], "g")
    plt.legend(["x", "v", "a"])
    plt.title("ERROR [log]")
    plt.show()

    plt.figure()
    plt.plot(time_span, x_est[:, 1], "r")
    plt.plot(time_span, v_est[:, 1], "b")
    plt.plot(time_span, a_est[:, 1], "g")
    plt.legend(["x", "v", "a"])
    plt.title("ESTIMATE")
    plt.show()

    plt.figure()
    plt.plot(time_span, x_real[:, 1], "r")
    plt.plot(time_span, v_real[:, 1], "b")
    plt.plot(time_span, a_real[:, 1], "g")
    plt.legend(["x", "v", "a"])
    plt.title("REAL")
    plt.show()
    exit()


def test_RK45():
    from .integrators import RK45

    def dynamics(t, s):
        K = 150
        x1 = s[0]
        x2 = s[1]
        dyn = np.array([
            x2,
            -2*np.sqrt(K)*x2 - K*x1
        ])
        return dyn

    s0 = np.ones(shape=(2,))
    s0[1] = 0
    t0 = 0
    dyn = lambda t, z: dynamics(t, z)

    time_span, z_span = RK45(dyn, t0, s0, h_init=10, T=3)

    plt.figure()
    plt.plot(time_span, z_span, "r")
    plt.show()


def test_potential():

    # obs, obs_U = obstacle_point_radial_static(o, 4, 1.5)

    xs = 1000
    x = np.linspace(-2, 2, xs)

    limit = 50
    o = np.array([0])
    beta = [0.25, 0.5, 1, 1.5]

    plt.figure()
    for b in beta:
        obs, obs_U = obstacle_point_radial_static(o, 1, b)
        zv = np.array([obs_U(xt, 0) for xt in x])
        zv[zv > limit] = limit
        plt.plot(x, zv)

    plt.ylim((-0.5, limit/2))
    plt.legend([str(b) for b in beta])
    plt.show()


def test_potential_3D():
    from matplotlib import cm

    o = np.array([0, 0])
    obs, obs_U = obstacle_point_dynamic(o, 2, 2)
    obs, obs_U = obstacle_point_static(o, 4)

    xs, ys = 200, 200
    v = np.array([3, 3])
    x = np.linspace(-2, 2, xs)
    y = np.linspace(-2, 2, ys)
    xv, yv = np.meshgrid(x, y)
    xv = xv.ravel()[:, np.newaxis]
    yv = yv.ravel()[:, np.newaxis]
    X = np.concatenate([xv, yv], axis=1)
    potential = np.array([obs_U(X[i, :], v) for i in range(X.shape[0])])
    xv = xv.reshape((xs, ys))
    yv = yv.reshape((xs, ys))
    zv = potential.reshape((xs, ys))

    limit = 50
    zv[zv > limit] = limit

    plt.figure()
    ax = plt.axes(projection="3d")
    ax.view_init(15, 15)
    # ax.view_init(0, 0)
    ax.plot_surface(xv, yv, zv, cmap=cm.coolwarm, linewidth=1, antialiased=True)
    ax.set_zlim((-0.1, limit/2))
    plt.show()


if __name__ == "__main__":
    # # VARIOUS TESTS
    # test_estimate_derivatives()
    # test_RK45()
    # test_potential()
    # test_potential_3D()
    # # TRAJECTORIES
    # test_default_trajectory()
    # test_simple_trajectory()
    # test_simple_3D_trajectory()
    # test_simple_trajectory_robustness(random_start=False)
    test_simple_trajectory_robustness(random_goal=False)
    # test_simple_trajectory_robustness()
    # test_ugly_trajectory()
    # test_ugly_3D_trajectory()
    # test_simple_3D_trajectory()
    # test_simple_scalable_trajectory()
    # test_simple_3D_scalable_trajectory()
    # test_scalability_moving_goal()
    # test_simple_3D_scalable_trajectory_moving_goal()
    # test_ugly_obstacles()
    # test_obstacles()
    # test_obstacles_3D_moving_goal()
