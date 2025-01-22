#!/usr/bin/env python3
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).absolute().parent.parent))

import rospy
import numpy as np

from yumi_controller.msg import YumiVelocity as YumiVelocityMsg

from core.controller_base import YumiDualController


class YumiVelocityController(YumiDualController):
    """ Class for running velocity control using an instance of `YumiDualController`.
        Trajectory parameters are sent with ROS Message `YumiTrajectory` and from 
        those a `YumiTrajectory` is constructed and tracked using the chosen 
        `YumiDualCartesianVelocityControlLaw`.
    """
    def __init__(self, velocity_topic: str):
        super().__init__(iksolver=self.IKSolver.PINV, symmetry=0.)
        
        # prepare velocity placeholder
        self.vel_r = []
        self.vel_l = []
        self.reset()
        
        # listen for velocity commands
        rospy.Subscriber(velocity_topic, YumiVelocityMsg, self._callback_velocity, queue_size=1, tcp_nodelay=False)
        
    def reset(self):
        pass
    
    def _callback_velocity(self, data: YumiVelocityMsg):
        """ Gets called when a new command is received. 
        """
        # update the trajectory
        self.vel_r = data.velocity_right
        self.vel_l = data.velocity_left
        print(f"New velocity received")
            
    def policy(self):
        """ Set target velocity for the current time step.
        """

        # SET VELOCITIES
        action = dict()
        # set velocities
        action["control_space"] = "individual"
        
        if self.vel_r != []:
            action["right_velocity"] = self.vel_r
            self.vel_r = []
        if self.vel_l != []:
            action["left_velocity"] = self.vel_l
            self.vel_l = []
        
        return action


class SimpleVelocityController(YumiVelocityController):
    def __init__(self):
        super().__init__("/desired_velocity")


def main():
    
    # starting ROS node
    rospy.init_node("velocity_controllers", anonymous=False)
    
    yumi_controller = SimpleVelocityController()
    
    def shutdown_callback():
        yumi_controller.pause()
        print("Controller shutting down")
    
    rospy.on_shutdown(shutdown_callback)
    
    
    yumi_controller.start()  # locking


if __name__ == "__main__":
    main()
