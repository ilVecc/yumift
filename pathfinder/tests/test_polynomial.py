import numpy as np, quaternion as quat

from .. import QuaternionParam, CubicQuatTrajectory, MultiParam, PoseParam, CubicPosePath
from ..visualization.plotter import plot_quat_mollweide, plot_quat_sphere
    
def test_main():
    
    vi = np.pi/2 * np.array([0, 0.707, 0.707])
    vf = np.pi/2 * np.array([0.707, 0.707, 0])
    
    traj = CubicQuatTrajectory()
    qi = QuaternionParam(quat.from_rotation_vector(vi), np.zeros(3))
    qf = QuaternionParam(quat.from_rotation_vector(vf), np.zeros(3))
    traj.update(qi, qf, tf=4)

    out = []
    for t in np.linspace(0, 4, 100, endpoint=True):
        param = traj.compute(t)
        out.append(param.quat)
    
    out = np.array(out)
    plot_quat_mollweide(out, "trajectory")


def test_position():
    return 


def test_quaternion():
    
    traj_1 = [
        (np.array([0.35, -0.20, 0.20]), quat.quaternion(0.7071067811865476, 0.7071067811865475, 0.0, 0.0)),
        (np.array([0.35, -0.10, 0.04]), quat.quaternion(0.35355339059327384, -0.8535533905932737, 0.14644660940672627, 0.3535533905932738)),
        (np.array([0.45, -0.15, 0.15]), quat.quaternion(0.7071067811865476, 0.0, 0.7071067811865475, 0.0)),
    ]
    
    traj_2 = [
        (np.array([0.35, +0.20, 0.20]), quat.quaternion(0.7071067811865476, 0.7071067811865475, 0.0, 0.0)),
        (np.array([0.35, +0.10, 0.04]), quat.quaternion(0.35355339059327384, 0.8535533905932737, 0.14644660940672627, -0.3535533905932738)),
        (np.array([0.45, +0.15, 0.15]), quat.quaternion(0.7071067811865476, 0.0, 0.7071067811865475, 0.0)),
    ]
    
    time = [0, 5, 5]
    traj = CubicPosePath()
    traj.update([
        MultiParam[PoseParam](PoseParam(p, q, np.zeros(6)), t) 
        for t, (p, q) in zip(time, traj_2)
    ])

    T = np.linspace(0, sum(time), 100, endpoint=True)
    out = np.array([traj.compute(t).rot for t in T])
    
    # plot_quat_mollweide(out, "trajectory")
    plot_quat_sphere(out)


if __name__ == "__main__":
    # test_main()
    # test_position()
    test_quaternion()
