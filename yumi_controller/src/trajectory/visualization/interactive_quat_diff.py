import numpy as np
import quaternion as quat

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from .plotter import plot_quat, Arrow3D


fig = plt.figure()
fig.subplots_adjust(bottom=0.1, right=0.8)
ax = fig.add_subplot(projection='3d')
ax.set_xlim3d(-1, +1)
ax.set_ylim3d(-1, +1)
ax.set_zlim3d(-1, +1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=20., azim=-35)


class Quat:
    
    def __init__(self, w, x, y, z):
        self.q : np.quaternion = None
        self.qx : np.quaternion = None
        self.qy : np.quaternion = None
        self.qz : np.quaternion = None
        self.h = lambda _ : _
        self.update(w, x, y, z, remove=False)

    def update(self, w, x, y, z, remove=True):
        self.q = np.quaternion(w, x, y, z).normalized()
        if remove:
            self.qx.remove(); self.qy.remove(); self.qz.remove()
        self.qx, self.qy, self.qz = plot_quat(ax, self.q)
        plt.draw()
        self.h(self.q)
    
    def change_w(self, event):
        self.update(event, self.q.x, self.q.y, self.q.z)
        
    def change_x(self, event):
        self.update(self.q.w, event, self.q.y, self.q.z)
        
    def change_y(self, event):
        self.update(self.q.w, self.q.x, event, self.q.z)
        
    def change_z(self, event):
        self.update(self.q.w, self.q.x, self.q.y, event)
    
    def on_change(self, handler):
        self.h = handler


class QuatDiff:
    
    def __init__(self, q1: np.quaternion, q2: np.quaternion):
        self.q1 : np.quaternion
        self.q2 : np.quaternion
        self.qx = self.qy = self.qz = self.q_axis = None
        self.angle : float
        self.shortest : bool
        self.flip : bool
        self.update(q1, q2, angle=0, shortest=True, flip=False, remove=False)

    def update(self, q1, q2, angle, shortest, flip, remove=True):
        # calculate qr s.t. qr * q1 = q2
        if q1 is not None:
            self.q1 = q1
        if q2 is not None:
            self.q2 = q2
        self.angle = angle
        self.shortest = shortest
        self.flip = flip
        ### minimum rotation difference ###
        q1, q2 = self.q1, self.q2
        if self.shortest and quat.as_float_array(q1) @ quat.as_float_array(q2) < 0:
            q2 = -q2
        if self.flip:
            q2 = -q2
        qr = (q2 * q1.conj()).normalized()
        ###################################
        qr_axis = quat.as_rotation_vector(qr)
        qr_axis /= np.linalg.norm(qr_axis)
        q = quat.from_rotation_vector(self.angle * quat.as_rotation_vector(qr)) * self.q1
        if remove:
            self.qx.remove(); self.qy.remove(); self.qz.remove(); self.q_axis.remove()
        self.qx, self.qy, self.qz = plot_quat(ax, q, ["orange", "limegreen", "royalblue"])
        self.q_axis = ax.add_artist(Arrow3D.quiv(qr_axis, "black"))
        plt.draw()
    
    def change_q1(self, q1):
        self.update(q1, None, self.angle, self.shortest, self.flip)
        
    def change_q2(self, q2):
        self.update(None, q2, self.angle, self.shortest, self.flip)
    
    def change_angle(self, event):
        self.update(None, None, event, self.shortest, self.flip)
    
    def change_toggles(self, event):
        if event == "shortest":
            self.update(None, None, self.angle, not self.shortest, self.flip)
        if event == "flip":
            self.update(None, None, self.angle, self.shortest, not self.flip)


def make_quat(xc, yc, quat=[1, 0, 0, 0]):
    ax_w = fig.add_axes([xc, yc, 0.065, 0.05])  # x%,y%,w%,h%
    w = Slider(ax_w, "w", -1, +1, quat[0], valstep=0.1)
    ax_x = fig.add_axes([xc, yc - 0.05, 0.065, 0.05])
    x = Slider(ax_x, "x", -1, +1, quat[1], valstep=0.1)
    ax_y = fig.add_axes([xc, yc - 0.10, 0.065, 0.05])
    y = Slider(ax_y, "y", -1, +1, quat[2], valstep=0.1)
    ax_z = fig.add_axes([xc, yc - 0.15, 0.065, 0.05])
    z = Slider(ax_z, "z", -1, +1, quat[3], valstep=0.1)

    call = Quat(w.val, x.val, y.val, z.val)
    w.on_changed(call.change_w)
    x.on_changed(call.change_x)
    y.on_changed(call.change_y)
    z.on_changed(call.change_z)

    return call

call1 = make_quat(0.9, 0.8, [1, 1, 0, 0])
call2 = make_quat(0.9, 0.55, [1, -1, 0, 0])


callr = QuatDiff(call1.q, call2.q)
call1.on_change(callr.change_q1)
call2.on_change(callr.change_q2)


ax_angle = fig.add_axes([0.05, 0.05, 0.85, 0.05])
angle = Slider(ax_angle, "angle", 0, 1, 0, valstep=0.02)
angle.on_changed(callr.change_angle)
ax_shortest = fig.add_axes([0.90, 0.05, 0.05, 0.05])
shortest = CheckButtons(ax_shortest, ["shortest", "flip"], [True, False])
shortest.on_clicked(callr.change_toggles)


plt.show()