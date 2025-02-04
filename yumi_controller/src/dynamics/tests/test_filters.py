import numpy as np
import matplotlib.pyplot as plt

from .. import *

def make_noisy_step(vals: dict, dims: int = 3, h: float = 0.001, T: float = 2.5, mu: float = 0, sigma: float = 0.025):
    t = np.linspace(0, T, int(1/h), endpoint=True)
    s = np.zeros((t.shape[0], dims))
    for ti, vi in vals.items():
        s[t >= ti, ...] = vi
    s += np.random.normal(mu, sigma, size=s.shape)
    return t, s


def test_admittance_tustin():
    """ Test for a 3D admittance with direct Tustin discretization. 
    """
    
    m = 2  # we can use floats ...
    k = np.array([1000, 5000, 700])  # ... or Numpy array (everything will be broadcasted to the right shape)
    k = 1000
    d = 2*np.sqrt(m*k)  # for a critically damped admittance
    h = 1/1000  # sampling step [s]
    
    adm = AdmittanceTustin(m, k, d, h)  # d=None yields the same result
    
    # three noisy box signals sampled with step h
    # each will pass through a different admittance
    t, u = make_noisy_step({0.5: [25, 15, 10], 1.5: 0}, h=h)
    
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

def test_lpfilter():
    """ Test for a 3D LP filter with direct Tustin discretization. 
    """
    
    f = 5
    k = 1
    h = 1/1000  # sampling step [s]
    
    lp = LPFilterTustin(f, k, h)
    
    # sine wave signal sampled with step h
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

def test_admittance_force():
    """ Test for a 3D force admittance with Tustin discretization of the state 
        space representation. 
    """
    
    m = 2
    k = 500  # or `np.diag([800, 1000, 1200])` for different values
    d = None
    h = 1/1000  # sampling step [s]
    adm = AdmittanceForce(m, k, d, h, method="tustin")
    
    # three noisy box signals sampled with step h
    # each will pass through a different admittance
    t, force = make_noisy_step({0.5: [25, 0, 0], 1.5: 0}, h=h)
    
    # calculate the output signal
    p, dp = adm.compute_signal(force)
    
    # plot the result
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(t, force, label=r"$u$")
    ax[0].legend()
    ax[1].plot(t, p, linestyle="-", label=r"$y$")
    ax[1].plot(t, dp, linestyle="--", label=r"$\dot{y}$")
    ax[1].legend()
    plt.show()

def test_admittance_torque():
    """ Test for a 3D torque admittance with Tustin discretization of the state 
        space representation. 
    """
    
    h = 1/1000  # sampling step [s]
    adm = AdmittanceTorque(m=0.001, d=None, k=0.2, h=h, method="forward")
    
    # three noisy signals sampled with step h
    # each will pass through a different admittance
    t, torque = make_noisy_step(h=h, T=5.0, sigma=0.01, vals={
        0.5: [0.5, 0.5, 0], 
        1.0: [0.5, 0.25, 0],
        1.5: 0})
    
    # calculate the output signal
    Q, W = adm.compute_signal(torque)
    
    # plot the result
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(t, torque)
    ax[0].legend(["$\\tau_x$", "$\\tau_y$", "$\\tau_z$"])
    ax[1].plot(t, quat.as_float_array(Q), linestyle="-")
    ax[1].legend(["$q_s$", "$q_x$", "$q_y$", "$q_z$"])
    plt.show()

def test_admittance_torque_anim():
    """ Test for a 3D torque admittance with Tustin discretization of the state 
        space representation. The plot is an animation in matplotlib.
    """
    
    import quaternion as quat
    from trajectory.MOVEME_plotter import animate_quaternion
    
    h = 1/1000  # sampling step [s]
    adm = AdmittanceTorque(m=0.001, d=0.02, k=0.0, h=h, method="forward")
    
    # three noisy box signals sampled with step h
    # each will pass through a different admittance
    t = np.linspace(0, 10.0, int(1/h), endpoint=True)
    torque = np.zeros((t.shape[0], 3))
    mu, sig = 2.5, 0.5
    torque[:, 0] = 0.45 * np.exp(-.5*((t - mu)/sig)**2)
    torque += np.random.normal(0, 0.01, size=torque.shape)
    
    # calculate the output signal
    Q, W = adm.compute_signal(torque)
    
    # animate the rotation
    fixed_frame = False
    base_rot = quat.quaternion(1/2,0,np.sqrt(3)/2,0)
    if fixed_frame:
        Q = quat.from_float_array(Q) * base_rot
    else:
        Q = base_rot * quat.from_float_array(Q)
    
    animate_quaternion(t, Q)

def test_admittance_timing():
    """ Same as `test_admittance_force` with a focus on timing.
    """
    
    import time
    
    h = 1/1000  # sampling step [s]
    adm_f = AdmittanceForce(m=1, d=None, k=100, h=h, method="tustin")
    adm_t = AdmittanceTorque(m=0.001, d=None, k=0.2, h=h, method="tustin")
    adm_w = AdmittanceWrench(m=np.diag(3*[1] + 3*[0.001]), d=None, k=np.diag(3*[100] + 3*[0.2]), h=h, method="tustin")
    
    # three noisy box signals
    t, force = make_noisy_step({0.5: [20, 0, 0], 1.5: 0}, h=h)
    t, torque = make_noisy_step({0.5: [0.5, 0.5, 0], 1.5: 0}, h=h, sigma=0.01)
    wrench = np.hstack([force, torque])
    
    # calculate the timing
    adm_f.clear()
    timing = []
    for fi in force:
        init = time.time()
        _, _  = adm_f.compute(fi)
        elap = time.time() - init
        timing.append(elap)
    mean = np.mean(timing)
    print(f"AdmittanceForce  avg compute time:                         {mean:.10f} s  (runs at {1/mean:6.4f} Hz)")
    
    adm_f.clear()
    timing = []
    for fi in force:
        init = time.time()
        _, _  = adm_f.compute(fi, h)
        elap = time.time() - init
        timing.append(elap)
    mean = np.mean(timing)
    print(f"AdmittanceForce  avg compute time (with h recomputation):  {mean:.10f} s  (runs at {1/mean:6.4f} Hz)")
    
    adm_t.clear()
    timing = []
    for ti in torque:
        init = time.time()
        _, _ = adm_t.compute(ti)
        elap = time.time() - init
        timing.append(elap)
    mean = np.mean(timing)
    print(f"AdmittanceTorque avg compute time:                         {mean:.10f} s  (runs at {1/mean:6.4f} Hz)")
    
    adm_t.clear()
    timing = []
    for ti in torque:
        init = time.time()
        _, _ = adm_t.compute(ti, h)
        elap = time.time() - init
        timing.append(elap)
    mean = np.mean(timing)
    print(f"AdmittanceTorque avg compute time (with h recomputation):  {mean:.10f} s  (runs at {1/mean:6.4f} Hz)")
    
    adm_w.clear()
    timing = []
    for wi in wrench:
        init = time.time()
        _, _ = adm_w.compute(wi)
        elap = time.time() - init
        timing.append(elap)
    mean = np.mean(timing)
    print(f"AdmittanceWrench avg compute time:                         {mean:.10f} s  (runs at {1/mean:6.4f} Hz)")
    
    timing = []
    for wi in wrench:
        init = time.time()
        _, _ = adm_w.compute(wi, h)
        elap = time.time() - init
        timing.append(elap)
    mean = np.mean(timing)
    print(f"AdmittanceWrench avg compute time (with h recomputation):  {mean:.10f} s  (runs at {1/mean:6.4f} Hz)")
    
    adm_f.clear()
    adm_t.clear()
    timing = []
    for (fi, ti) in zip(force, torque):
        init = time.time()
        _, _ = adm_f.compute(fi)
        _, _ = adm_t.compute(ti)
        elap = time.time() - init
        timing.append(elap)
    mean = np.mean(timing)
    print(f"Admittance(F+T)  avg compute time:                         {mean:.10f} s  (runs at {1/mean:6.4f} Hz)")
    
    adm_f.clear()
    adm_t.clear()
    timing = []
    for (fi, ti) in zip(force, torque):
        init = time.time()
        _, _ = adm_f.compute(fi, h)
        _, _ = adm_t.compute(ti, h)
        elap = time.time() - init
        timing.append(elap)
    mean = np.mean(timing)
    print(f"Admittance(F+T)  avg compute time (with h recomputation):  {mean:.10f} s  (runs at {1/mean:6.4f} Hz)")


if __name__ == "__main__":
    test_admittance_timing()