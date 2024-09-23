try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import numpy as np
import quaternion as quat


def make_quaternion(az, el, th):
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


if __name__ == "__main__":
    
    out = np.array([
        quat.from_rotation_vector( np.pi/3 * np.array([1, 0, 0])),
        quat.from_rotation_vector( np.pi/2 * np.array([0.707, 0.707, 0]))
    ])
    plot_quat(out, "collection")