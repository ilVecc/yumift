#!/usr/bin/env python3
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).absolute().parent.parent))

import rospy
import numpy as np

from core.controller_base import YumiDualController, YumiDualDeviceState
from core.control_laws import YumiDualAdmittanceControlLaw
from core.parameters import Parameters

from gains import GAINS


class YumiLeadController(YumiDualController):
    """ Class for running lead-through control using an instance of `YumiDualController`.
    """
    def __init__(self):
        super().__init__()
        self.current_state.alpha = 0.5
        self.debug = True
        
        # define control law
        # TODO maybe set M=??, D=??, K=0
        self.control_law = YumiDualAdmittanceControlLaw(GAINS, "forward")
        self.effective_mode = "individual"
        
        self.reset(state)  # init trajectory (set current position)

    def reset(self, state: YumiDualDeviceState):
        """ Initialize the controller setting the current point as desired trajectory. 
        """
        self.control_law.clear()
        self.control_law.mode = "individual"
        print("Controller reset")
    
    def policy(self):
        """ Calculate target velocity for the current time step.
        """
        
        # update timing information
        now = rospy.Time.now()
        dt = (now - self.current_state.timestamp).to_sec()
        self.control_law.update_current_dt(dt)
        
        # update pose and wrench for the control law class
        self.control_law.update_current_state(self.current_state)
        
        # calculate new target velocities and positions for this time step
        self.control_law.update_desired_state(self.current_state)
        
        # CALCULATE VELOCITIES
        action = dict()
        # set velocities based on control mode
        try:
            # get space based on control mode ...
            action["control_space"] = self.control_law.mode
            action["timestep"] = dt
            vel_1, vel_2 = self.control_law.compute_target_state()
            # ... but use the effective mode to set the velocities
            if self.effective_mode == "individual":
                action["right_velocity"], action["left_velocity"] = vel_1, vel_2
            elif self.effective_mode == "right":
                action["right_velocity"] = vel_1
            elif self.effective_mode == "left":
                action["left_velocity"] = vel_2
            elif self.effective_mode == "coordinated":
                action["absolute_velocity"], action["relative_velocity"] = vel_1, vel_2
            elif self.effective_mode == "absolute":
                action["absolute_velocity"] = vel_1
            elif self.effective_mode == "relative":
                action["relative_velocity"] = vel_2
                
        except Exception as ex:
            print(f"Stopping motion (exception: {ex})")
            action = {
                "control_space": "joint_space",
                "joint_velocities": np.zeros(Parameters.dof)}
        
        return action


def main():
    # starting ROS node
    rospy.init_node("lead_controller", anonymous=False) 
    
    yumi_controller = YumiLeadController()
    
    def shutdown_callback():
        yumi_controller.pause()
        print("Controller shutting down")
    
    rospy.on_shutdown(shutdown_callback)
    
    yumi_controller.start()


if __name__ == "__main__":
    main()
