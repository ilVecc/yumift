try:
    from typing import Literal, Union
except ImportError:
    from typing_extensions import Literal, Union

import numpy as np, quaternion as quat

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


#
# Mollweide projection plot for quaternion trajectory
#

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
    v_norm = np.linalg.norm(np.asarray(v), axis=1)
    w = v / (v_norm + (v_norm == 0))[:, np.newaxis]
    if return_norm:
        return w, v_norm
    return w


def plot_quat_mollweide(q, format: Literal["trajectory", "trajectory+", "collection"] = "trajectory"):
    rv, ra = normalize(quat.as_rotation_vector(q), return_norm=True)

    lon = np.arctan2(rv[:, 1], rv[:, 0])
    lat = np.arcsin(rv[:, 2])
    theta = ra

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

def plot_quat_sphere(q):
    rv, ra = normalize(quat.as_rotation_vector(q), return_norm=True)
    
    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(projection="3d")
    ax.grid(True)
    # ax.set_aspect('equal')
    
    cmap = mpl.cm.get_cmap("inferno")
    
    # origin
    plot_quat(ax, quat.one, length=0.5)
    
    # sphere wireframe
    radius = 1
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color="k", alpha=0.1)
    
    # quaternion trajectory
    c = [cmap(a/np.pi) for a in ra]
    ax.scatter(rv[:,0], rv[:,1], rv[:,2], c=c)
    for i in range(rv.shape[0]-1):
        ax.plot(rv[i:i+2,0], rv[i:i+2,1], rv[i:i+2,2], color=c[i])
    
    # frenet frames on sphere
    plot_traj_quat(ax, q, alpha=0.2)
    
    # axel angle colorbar
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(0, np.pi))
    cbar = fig.colorbar(sm, ax=ax, location="right")
    cbar.set_ticks(np.linspace(0, np.pi, 13, endpoint=True))
    cbar.set_ticklabels(list(map(str, range(0, 181, 15))))
    
    plt.show()


#
# Frenet frame plot for SE(3) trajectory
#

def plot_traj_pose(ax : Union[plt.Axes, None], pos: np.ndarray, rot: np.ndarray, alpha : float = 1.0, length : float = 0.1):
    
    rotmat = quat.as_rotation_matrix(rot)
    
    if ax is None:
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

def plot_traj_quat(ax : Union[plt.Axes, None], Q : np.ndarray, alpha : float = 1.0, length : float = 0.1):
    
    if ax is None:
        ax = plt.figure().add_subplot(projection='3d')
    
    rotmat = quat.as_rotation_matrix(Q)
    pos, _ = normalize(quat.as_rotation_vector(Q), return_norm=True)
    
    ax.quiver(pos[:, 0], pos[:, 1], pos[:, 2], rotmat[:, 0, 0], rotmat[:, 1, 0], rotmat[:, 2, 0], length=length, colors="r", alpha=alpha)
    ax.quiver(pos[:, 0], pos[:, 1], pos[:, 2], rotmat[:, 0, 1], rotmat[:, 1, 1], rotmat[:, 2, 1], length=length, colors="g", alpha=alpha)
    ax.quiver(pos[:, 0], pos[:, 1], pos[:, 2], rotmat[:, 0, 2], rotmat[:, 1, 2], rotmat[:, 2, 2], length=length, colors="b", alpha=alpha)
    

#
# Matplotlib animation for quaternion trajectory
#

# adapted from https://stackoverflow.com/questions/63546097/3d-curved-arrow-in-python
class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, u, v, w, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = [x, u], [y, v], [z, w]

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)
    
    @staticmethod
    def quiv(v, color="r", o=[0,0,0], length=1.0):
        return Arrow3D(*o,*(v*length), mutation_scale=20, lw=1, arrowstyle="-|>", color=color)  #, connectionstyle="arc3,rad=-0.3")

def plot_quat(
    ax: plt.Axes, 
    Q: Union[np.quaternion, np.ndarray], 
    colors: list = ["r", "g", "b"],
    origin : np.ndarray = np.zeros(3),
    length : float = 1.0
):
    """ Plot a quaternion as a reference frame
    :param ax: the `figure.Axes` to use
    :param Q: the quaternion to plot (takes priority over `rotmat`)
    :param colors: a list of three color for the axes (in order, x, y, and z)
    :param origin: the origin of the quaternion, by default (0,0,0)
    """
    if isinstance(Q, np.quaternion):
        rotmat = quat.as_rotation_matrix(Q)
    
    qx = ax.add_artist(Arrow3D.quiv(rotmat[:, 0], colors[0], origin, length))
    qy = ax.add_artist(Arrow3D.quiv(rotmat[:, 1], colors[1], origin, length))
    qz = ax.add_artist(Arrow3D.quiv(rotmat[:, 2], colors[2], origin, length))
    
    return [qx, qy, qz]


def animate_quaternion(t: np.ndarray, Q: np.ndarray):
    
    ax = plt.figure().add_subplot(projection='3d')
    ax.set_xlim3d(-1, +1)
    ax.set_ylim3d(-1, +1)
    ax.set_zlim3d(-1, +1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20., azim=-35)
    
    rotmat = quat.as_rotation_matrix(Q)
    
    # canonical base
    plot_quat(ax, quat.one, color=["orange", "limegreen", "royalblue"])
    # initial quaternion
    plot_quat(ax, rotmat[0], color=["magenta", "yellow", "cyan"])
    # moving quaternion
    qx, qy, qz = plot_quat(ax, rotmat[0], color=["r", "g", "b"])
    
    def update(frame):
        nonlocal qx, qy, qz, rotmat
        qx.remove(); qy.remove(); qz.remove()
        qx, qy, qz = plot_quat(ax, rotmat[frame], color=["r", "g", "b"])
    
    # WARNING: keep this unused assignment, animation doesn't start otherwise
    # TODO only accept uniform timing, this forced interval is very ugly
    ani = mpl.animation.FuncAnimation(fig=plt.gcf(), func=update, frames=len(t), interval=1/np.mean(np.diff(t)))
    plt.show()


def main():    
    # out = np.array([
    #     quat.from_rotation_vector( np.pi/3 * np.array([1, 0, 0])),
    #     quat.from_rotation_vector( np.pi/2 * np.array([0.707, 0.707, 0]))
    # ])
    # plot_quat(out, "collection")
    
    from pathfinder.polynomial import CubicPoseTrajectory, PoseParam
    
    vi = np.pi/2 * normalize([[0., 1., 1.]])[0]
    vf = np.pi/2 * normalize([[1., 1., 0.]])[0]
    
    tf = 4
    traj = CubicPoseTrajectory()
    pi = PoseParam(np.array([0., 0., 0.]), quat.from_rotation_vector(vi), np.zeros(6))
    pf = PoseParam(np.array([3., 2., 1.]), quat.from_rotation_vector(vf), np.zeros(6))
    traj.update(pi, pf, tf)

    time = np.linspace(0, tf, 100, endpoint=True)
    out = [traj.compute(t) for t in time]
    out_pos = np.array([p.pos for p in out])
    out_rot = np.array([p.rot for p in out])
    
    plot_traj_pose(out_pos, out_rot)


if __name__ == "__main__":
    main()