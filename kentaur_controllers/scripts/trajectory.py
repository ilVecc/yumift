import numpy as np
import quaternion as quat

from pathfinder.visualization.plotter import plot_traj_pose

class WhiteboardTrajectory():
    
    @staticmethod
    def yumi_sine_yz(time : float):
        time = min(max(0, time), 5)
        f = 0.5
        
        x = 0.020
        y = 0.050 - 0.050/5*time
        z = 0.035 + 0.040*np.sin(2*np.pi*f*time)
        ori = quat.from_euler_angles(0, np.pi/2, -np.pi/2)
        
        return np.array([x, y, z]), ori
    

if __name__ == "__main__":
    time = np.linspace(0, 5, 200, endpoint=True)
    
    pos, ori = map(np.asarray, zip(*[WhiteboardTrajectory.yumi_sine_yz(t) for t in time]))
    
    plot_traj_pose(None, pos, ori, scale=0.01)
    