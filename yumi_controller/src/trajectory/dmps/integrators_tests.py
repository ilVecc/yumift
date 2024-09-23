import numpy as np
import matplotlib.pyplot as plt 

from integrators import RK45, EUL

def test_RK45_vanderpol():
    
    # Van der Pol oscillator
    def vanderpol(t, x):
        mu = 2
        x1, x2 = x[0], x[1]
        dx = np.array([
            x2,
            -x1 + mu * (1 - x1**2) * x2
        ])
        return dx

    dyn = lambda t, z: vanderpol(t, z)
    solver = RK45(h_init=0.002)
    
    plt.figure()
    for _ in range(100):
        x0 = np.random.normal(0, 4, (2,)) #np.array([0, 0.5])
        time_span, x_span = solver(dyn, x0, t0=0, tf=20)
        plt.plot(x_span[:, 0], x_span[:, 1])
    
    plt.gca().set_aspect("equal", adjustable="datalim")
    plt.show()

def test_RK45_particles_stable():
    
    def particles(t, z):
        I = np.eye(2)
        Ih = 0.5*I
        dx = np.block([[-I, Ih, Ih],
                       [Ih, -I, Ih],
                       [Ih, Ih, -I]]) @ z
        return dx

    dyn = lambda t, z: particles(t, z)
    solver = RK45(h_init=0.002)
    
    plt.figure()
    z0 = np.random.uniform(-2., +2., (6,))
    time_span, z_span = solver(dyn, z0, t0=0, tf=20)
    for i in range(3):
        plt.plot(z_span[:, 2*i], z_span[:, 2*i+1])
    
    plt.gca().set_aspect("equal", adjustable="datalim")
    plt.show()

def test_RK45_particles_unstable_convergent():
    
    def particles(t, z):
        I = np.eye(2)
        O = np.zeros((2,2))
        Ih = 0.5*I
        dx = np.block([[-I, Ih, -I,  O,  O],
                       [ O, -I, Ih, -I,  O],
                       [ O,  O, -I, Ih, -I],
                       [-I,  O,  O, -I, Ih],
                       [Ih, -I,  O,  O, -I]]) @ z
        return dx
    
    dyn = lambda t, z: particles(t, z)
    solver = RK45(h_init=0.002)
    
    plt.figure()
    z0 = 2*np.concatenate([[np.cos(i), np.sin(i)] for i in np.linspace(0, 2*np.pi, 5, endpoint=False)])
    time_span, z_span = solver(dyn, z0, t0=0, tf=300)
    for i in range(5):
        plt.plot(z_span[:, 2*i], z_span[:, 2*i+1])
    
    plt.gca().set_aspect("equal", adjustable="datalim")
    plt.show()

def test_RK45_particles_unstable_divergent():

    def particles(t, z):
        I = np.eye(2)
        O = np.zeros((2,2))
        Ih = 2*I
        dx = np.block([[-I, -I, Ih,  O,  O],
                       [ O, -I, -I, Ih,  O],
                       [ O,  O, -I, -I, Ih],
                       [Ih,  O,  O, -I, -I],
                       [-I, Ih,  O,  O, -I]]) @ z
        return dx
    
    dyn = lambda t, z: particles(t, z)
    solver = RK45(h_init=0.002, h_min=1e-20)
    
    plt.figure()
    z0 = 2*np.concatenate([[np.cos(i), np.sin(i)] for i in np.linspace(0, 2*np.pi, 5, endpoint=False)])
    time_span, z_span = solver(dyn, z0, t0=0, tf=95)
    for i in range(5):
        plt.plot(z_span[:, 2*i], z_span[:, 2*i+1])
    
    plt.gca().set_aspect("equal", adjustable="datalim")
    plt.show()


if __name__ == "__main__":
    # test_RK45_vanderpol()
    # test_RK45_particles_stable()
    # test_RK45_particles_unstable_convergent()
    test_RK45_particles_unstable_divergent()
