try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import numpy as np
import quaternion as quat


def azel_to_quat(az, el, th):
    v = np.array([
        np.cos(el)*np.cos(az),
        np.cos(el)*np.sin(az),
        np.sin(el)
    ])
    return quat.from_rotation_vector(th*v)

def normalize(v, return_norm=False):
    """ Calculates the normalized vector
        :param v: np.array()
    """
    v_norm = np.linalg.norm(v, axis=1)
    w = v / (v_norm + (v_norm == 0))[:, np.newaxis]
    if return_norm:
        return w, v_norm
    return w

def plot_quat(q, format: Literal["trajectory", "trajectory+", "collection"] = "trajectory"):
    v, a = normalize(quat.as_rotation_vector(q), return_norm=True)

    lon = np.arctan2(v[:, 1], v[:, 0])
    lat = np.arcsin(v[:, 2])
    theta = a

    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(projection="mollweide")
    ax.grid(True)
    cmap = "inferno"
    
    if format.startswith("trajectory"):
        points = np.array([lon, lat]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1, ...], points[1:, ...]], axis=1)
        dist = np.linalg.norm(segments[:, 0, :] - segments[:, 1, :], axis=1)
        linewidths = 3 * (dist < 5)
        if format.endswith("+"):
            linewidths += 1 * (dist >= 5)
        lc = LineCollection(segments, array=theta, linewidth=linewidths, cmap=cmap, clim=(0, np.pi))
        pc = ax.add_collection(lc)
    elif format == "collection":
        pc = ax.scatter(lon, lat, c=theta, cmap=cmap, vmin=0, vmax=np.pi)

    cbar = fig.colorbar(pc, ax=ax, location="bottom")
    cbar.set_ticks(np.linspace(0, np.pi, 13, endpoint=True))
    cbar.set_ticklabels(list(map(str, range(0, 181, 15))))

    plt.show()
    
    
def plot_traj(pos: np.ndarray, rot: np.ndarray):
    
    rotmat = quat.as_rotation_matrix(rot)

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], c="k", lw=10)
    ax.quiver(pos[:, 0], pos[:, 1], pos[:, 2], rotmat[:, 0, 0], rotmat[:, 1, 0], rotmat[:, 2, 0], length=0.1, colors="r")
    ax.quiver(pos[:, 0], pos[:, 1], pos[:, 2], rotmat[:, 0, 1], rotmat[:, 1, 1], rotmat[:, 2, 1], length=0.1, colors="g")
    ax.quiver(pos[:, 0], pos[:, 1], pos[:, 2], rotmat[:, 0, 2], rotmat[:, 1, 2], rotmat[:, 2, 2], length=0.1, colors="b")
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20., azim=-35)

    plt.show()


if __name__ == "__main__":
    
    # out = np.array([
    #     quat.from_rotation_vector( np.pi/3 * np.array([1, 0, 0])),
    #     quat.from_rotation_vector( np.pi/2 * np.array([0.707, 0.707, 0]))
    # ])
    # plot_quat(out, "collection")
    
    import sys, pathlib; sys.path.append(str(pathlib.Path(__file__).parent.parent))
    from trajectory.polynomial import CubicPoseTrajectory, PoseParam
    
    vi = np.pi/2 * np.array([0, 0.707, 0.707])
    vf = np.pi/2 * np.array([0.707, 0.707, 0])
    
    traj = CubicPoseTrajectory()
    pi = PoseParam(np.array([0., 0., 0.]), quat.from_rotation_vector(vi), np.zeros(6))
    pf = PoseParam(np.array([3., 2., 1.]), quat.from_rotation_vector(vf), np.zeros(6))
    traj.update(pi, pf, tf=4)

    out_pos = []
    out_rot = []
    for t in np.linspace(0, 4, 100, endpoint=True):
        param = traj.compute(t)
        out_pos.append(param.pos)
        out_rot.append(param.rot)
    out_pos = np.array(out_pos)
    out_rot = np.array(out_rot)
    
    plot_traj(out_pos, out_rot)