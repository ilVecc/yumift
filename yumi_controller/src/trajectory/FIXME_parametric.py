import rospy
import tf as ros_tf
import numpy as np

from typing import List


class CircularTrajectory(object):
    """ Generates a trajectory from trajectory parameters, the process is identical for individual and coordinated manipulation.
        The variable names follows individual motion with left and right. This means when coordinate manipulation is used, right
        is absolute motion and left becomes relative motion. """
    def __init__(self, origin, normal, radius, deltaTime):
        self.length = 0
        self.dt = deltaTime
        
        self._segment_index = 1  # current target trajectory parameter, 0 is the current position.
        self._segment_time = 0  # keeps track on time, resets for every trajectory segment
        self._segment_new = True
        
        self.position_velocities = []  # stores velocity transitions between trajectory parameters
        
        self.target_position = np.zeros(3)  # corresponding target position
        self.target_rotation = np.zeros(4)  # ... and orientation
        self.target_velocity = np.zeros(6)  # desired velocity from trajectory
        
        self._cache_rot_mats = []  # stores rotation matrices to save some computation
        self._transformer = ros_tf.TransformerROS(True, rospy.Duration(1.0))

    def update_trajectory(self, points: List[TrajectoryPoint], vel_init: float):
        """ updates the trajectory when a new trajectory as been received
            :param points: list of trajectory points,
            :param vel_init: np.array() shape(3) initial position velocity
        """
        # reset parameters and cache
        self.points = points
        self.length = len(self.points)
        self._segment_index = 1 
        self._segment_time = 0 
        self._segment_new = True
        
        # calculate list of rotation matrices for each trajectory parameters
        self._cache_rot_mats = []  # reset rotation matrices
        for i in range(self.length):
            mat = self._transformer.fromTranslationRotation(np.zeros(3), points[i].orientation)
            self._cache_rot_mats.append(mat[0:3, 0:3])

        # calculate the transition velocities between the trajectory parameters
        self.position_velocities = [vel_init]  # current velocity of the robot
        for i in range(1, self.length-1):
            vel = calc_point_vel(
                points[i-1].position, points[i].position, points[i+1].position, 
                points[i].duration, points[i+1].duration)
            self.position_velocities.append(vel)
        self.position_velocities.append(np.zeros(3))  # last point always has velocity 0

    def _update_segment(self):
        """ Updates the current target trajectory parameters or which is the 
            current segment on the trajectory.
        """
        if self._segment_time > self.points[self._segment_index].duration:
            if self._segment_index < self.length - 1:
                self._segment_time = 0
                self._segment_index += 1
                self._segment_new = True
            else:  
                # for last point
                self._segment_time = self.points[self._segment_index].duration
                self._segment_index = self.length - 1
                self._segment_new = False
        else:
            self._segment_new = False

    def is_new_segment(self):
        """ Returns True if a new segment has been entered, only shows true
            once for each segment. Used to check if new gripper commands should be sent.
        """
        return self._segment_new

    def get_next_target(self):
        """ calculates the desired velocities and target pose. """

        self._update_segment()
        
        # calculates the target position and desired velocity for a time point
        q, dq = calc_pos_vel(
            qi=self.points[self._segment_index-1].position, dqi=self.position_velocities[self._segment_index-1],
            qf=self.points[self._segment_index].position, dqf=self.position_velocities[self._segment_index],
            tf=self.points[self._segment_index].duration,
            t=self._segment_time)
        self.target_position = q
        self.target_velocity[0:3] = dq

        # calculates the target orientation and desierd angular velocity for a time point
        quat, we = calc_orientation(
            Ri=self._cache_rot_mats[self._segment_index-1],
            Rf=self._cache_rot_mats[self._segment_index],
            tf=self.points[self._segment_index].duration,
            t=self._segment_time)
        self.target_rotation = quat
        self.target_velocity[3:6] = we

        # update time 
        self._segment_time += self.dt

        return self.target_position, self.target_rotation, self.target_velocity
