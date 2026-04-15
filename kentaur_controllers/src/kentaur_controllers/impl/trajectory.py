import numpy as np
import quaternion as quat


class WhiteboardTrajectory():
    
    @staticmethod
    def yumi_sine_yz(time : float):
        time = min(max(0, time), 5)
        f = 0.5
        
        x = 0.020
        y = 0.050 - 0.010*time
        z = np.sin(2*np.pi*f*time)
        ori = quat.from_euler_angles(0, np.pi/2, -np.pi/2)
        
        return np.array([x, y, z]), ori
    