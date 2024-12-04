import numpy as np
import quaternion as quat

from .. import QuaternionParam, CubicQuatTrajectory
from ..MOVEME_plotter import plot_quat_mollweide
    
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
    
if __name__ == "__main__":
    test_main()