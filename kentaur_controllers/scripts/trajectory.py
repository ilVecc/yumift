import numpy as np
import quaternion as quat

from dynamicals.utils import Frame
from pathfinder.visualization.plotter import plot_traj_pose

class WhiteboardTrajectory():
    
    @staticmethod
    def task_to_world():
        return Frame([0, 0, 0.030], quat.quaternion(0.5,0.5,0,0))
    
    @staticmethod
    def task_trajectory(time : float):
        tmax = 5
        time = min(max(0, time), tmax)
        f = 0.5  # Hz
        x0, y0 = 0.050, 0.050
        vx = 0.150 / tmax  # 50mm in tmax
        
        x = x0 + vx * time
        y = y0 + 0.040*np.sin(2*np.pi*f*time)
        z = 0
        ori = quat.one
        
        return Frame([x, y, z], ori)
    
    @staticmethod
    def world_trajectory(time : float):
        return WhiteboardTrajectory.task_to_world() @ WhiteboardTrajectory.task_trajectory(time)


if __name__ == "__main__":
    time = np.linspace(0, 5, 200, endpoint=True)
    
    traj_frames = [WhiteboardTrajectory.world_trajectory(t) for t in time]
    pos, ori = map(np.array, zip(*[(f.pos, f.rot) for f in traj_frames]))
    
    plot_traj_pose(None, pos, ori, scale=0.005)
    