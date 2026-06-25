#!/usr/bin/env python3
from typing import List

import numpy as np
import quaternion as quat

from dynamicals.utils import Frame
from pathfinder.visualization.plotter import plot_traj_pose


class WhiteboardTrajectory():
    
    @staticmethod
    def task_to_world():
        ori = (quat.quaternion(1, 0, 0, 1)*quat.quaternion(1, 1, 0, 0)).normalized()
        return Frame([-0.850, 0, 0.800], ori)
    
    @staticmethod
    def task_trajectory(time : float):
        T = 10
        time = min(max(0, time), T)
        x0, y0 = 0.250, 0.050  # initial point
        vx = 1.000 / T  # 1000mm in tmax
        f = 0.5  # Hz
        a = 0.30  # amplitude
        # phi = np.pi/12  # phase
        phi = 0
        
        w = 2*np.pi*f
        
        x = x0 + vx * time
        y = y0 + a*np.sin(w*time + phi)
        z = 0
        
        # t = np.array([1, 0, 0])
        t = np.array([vx, a*w*np.cos(w*time + phi), 0])
        t /= np.linalg.norm(t)
        b = np.array([0,0,-1])
        n = np.cross(b, t)
        R = np.zeros((3,3))
        R[:,0] = t
        R[:,1] = n
        R[:,2] = b
        ori = quat.from_rotation_matrix(R)
        
        return Frame([x, y, z], ori)
    
    @staticmethod
    def world_trajectory(time : float):
        return WhiteboardTrajectory.task_to_world() @ WhiteboardTrajectory.task_trajectory(time) @ Frame(rotation=quat.quaternion(0,0,0,1).normalized())


def main_plot():
    time = np.linspace(0, 10, 200, endpoint=True)
    traj_frames = [WhiteboardTrajectory.world_trajectory(t) for t in time]
    
    pos, ori = map(np.array, zip(*[(f.pos, f.rot) for f in traj_frames]))
    plot_traj_pose(None, pos, ori, scale=0.005)


def fill_velocities(traj_frames : List[Frame], dt : float):
    from yumift_controllers.impl.trajectory import YumiTrajectory
    if traj_frames[0].vel is None:
        traj_frames[0].vel = np.zeros(6)
    for i in range(1, len(traj_frames)-1):
        if traj_frames[i].vel is None:
            traj_frames[i].vel = np.zeros(6)
            traj_frames[i].vel[0:3] = YumiTrajectory._calculate_intermediate_velocity_linear(
                traj_frames[i-1].pos, traj_frames[i].pos, traj_frames[i+1].pos, dt, dt)
            traj_frames[i].vel[3:6] = YumiTrajectory._calculate_intermediate_velocity_angular(
                traj_frames[i-1].rot, traj_frames[i].rot, traj_frames[i+1].rot, dt, dt)
    if traj_frames[-1].vel is None:
        traj_frames[-1].vel = np.zeros(6)
    return traj_frames

def main_send_postures():
    import rospy
    from yumift_msgs.helper import Helper as H
    from yumift_msgs.msg import YumiPosture
    
    rospy.init_node("demo_whiteboard_trajectory", anonymous=False)
    pub = rospy.Publisher("/posture", YumiPosture, queue_size=1)
    rospy.sleep(0.1)
    
    T = 10
    n = 1000
    dt = T/n
    time = np.linspace(0, T, n, endpoint=True)
    traj_frames = [WhiteboardTrajectory.world_trajectory(t) for t in time]
    traj_frames = fill_velocities(traj_frames, dt)
    
    diff = np.array([0, -0.50, 0])
    for i in range(len(traj_frames)):
        frame_r = traj_frames[i]
        pos = frame_r.pos
        ori = quat.as_float_array(frame_r.rot)
        vel = frame_r.vel
        pub.publish(H.posture(0, (pos, ori, vel ,0), (pos + diff, ori, vel, 0)))
        rospy.loginfo(f"Sent posture {i+1}/{len(time)}")
        rospy.sleep(dt)
    
    
if __name__ == "__main__":
    main_send_postures()
    # main_plot()
    