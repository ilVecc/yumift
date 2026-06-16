#!/usr/bin/env python3
import numpy as np
import quaternion as quat

from dynamicals.utils import Frame
from pathfinder.visualization.plotter import plot_traj_pose


class WhiteboardTrajectory():
    
    @staticmethod
    def task_to_world():
        return Frame([0, 0, 0.030], quat.z*quat.quaternion(0.5, 0.5, 0, 0))
    
    @staticmethod
    def task_trajectory(time : float):
        T = 10
        time = min(max(0, time), T)
        x0, y0 = 0.050, 0.050  # initial point
        x0, y0 = 0.0, 0.0  # initial point
        vx = 1.000 / T  # 1000mm in tmax
        f = 0.5  # Hz
        a = 0.040  # amplitude
        phi = np.pi/12  # phase
        
        w = 2*np.pi*f
        
        x = x0 + vx * time
        y = y0 + a*np.sin(w*time + phi)
        z = 0
        
        t = np.array([vx, a*w*np.cos(w*time + phi), 0])
        t /= np.linalg.norm(t)
        b = np.array([0,0,-1])
        R = np.zeros((3,3))
        R[:,0] = t
        R[:,1] = np.cross(b, t)
        R[:,2] = b
        ori = quat.from_rotation_matrix(R)
        
        return Frame([x, y, z], ori)
    
    @staticmethod
    def world_trajectory(time : float):
        return WhiteboardTrajectory.task_to_world() @ WhiteboardTrajectory.task_trajectory(time)


if __name__ == "__main__":
    time = np.linspace(0, 10, 200, endpoint=True)
    traj_frames = [WhiteboardTrajectory.world_trajectory(t) for t in time]
    
    pos, ori = map(np.array, zip(*[(f.pos, f.rot) for f in traj_frames]))
    plot_traj_pose(None, pos, ori, scale=0.005)
    