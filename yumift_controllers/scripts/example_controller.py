#!/usr/bin/env python3
import rospy
import numpy as np

from yumift_controllers.common.device import YumiDevice, YumiDualDeviceState
from yumift_controllers.common.controller_base import YumiDualController, MixedVelocityYumiAction
from yumift_controllers.common.control_laws import YumiIndividualCartesianVelocityControlLaw
from yumift_controllers.ik.algorithms import PINVIKAlgorithm
from yumift_controllers.impl.trajectory import YumiParam
from yumift_controllers.misc.utils import load_config, YumiCoordinatedRobotState_from_YumiParam


class YumiDummyController(YumiDualController):
    """ Dummy controller that uses the `YumiDualController` class.
        A constant posture is sent with a `YumiPosture` ROS Message and tracked 
        using the `YumiIndividualCartesianVelocityControlLaw` control law.
    """
    def __init__(self, device : YumiDevice):
        # Here the controller is initialized by setting its inverse kinematics
        # solver; in this example, the pseudo-inverse is chosen.
        super().__init__(device, ikalgorithms=[PINVIKAlgorithm()])
        
        # Define the control law computing velocity commands.
        # Here a cartesian velocity control law is showcased; the "individual"
        # stands for individual arm control, opposite of coordinated arm control.
        # Other control laws might have different features.
        self.control_law = YumiIndividualCartesianVelocityControlLaw(load_config("gains.yaml")["CLIK"])
        
        # The controller simply sets Yumi on a fixed posture with default position
        # at (0.4, -0.2, 0.2) and (0.4, 0.2, 0.2) for the right and left grippers
        self.desired_posture = YumiParam(vel_r=np.zeros(6), vel_l=np.zeros(6))
        
    def reset(self, state: YumiDualDeviceState):
        # This function is called every time the controller disconnects from Yumi,
        # for example when EGM is stopped after an error occured in Yumi.
        # This this controller has no special requirements, nothing is done here.
        # You might need to reset the internal status in other controllers, for 
        # example in a trajectory controller that stores the trajectory, which you 
        # may want to delete after an error.
        print("Reset function working")
        
    def policy(self, state: YumiDualDeviceState) -> MixedVelocityYumiAction:
        # This is the main component of the controller. Here the control action 
        # is decided and computed, and in particular control laws are used. 
        # Keep in mind that all the steps illustrated here might be unnecessary 
        # for your specific case.
        
        # The three following functions can be called in any order for this 
        # specific control law; check if this holds for your case as well.
        
        # First, we translate the desired `YumiParam` posture in a `RobotState` 
        # object for the control law. Then, we pass it to the control law.
        yumi_desired_state = YumiCoordinatedRobotState_from_YumiParam(self.desired_posture)
        self.control_law.update_desired_state(yumi_desired_state)
        
        # Then, timestep is updated. This uses the internal `self.yumi_time` 
        # field, which is automatically updated at the frequency specified in 
        # `robot_state_updater.py`. 
        dt = (rospy.Time.now() - state.time).to_sec()
        self.control_law.update_current_timestep(dt)
        
        # Finally, the current state of Yumi is set inside the control law.
        # Again, the field `self.yumi_state` is automatically updated.
        self.control_law.update_current_state(state)
        
        # Now, we can compute the required action that brings the current state
        # to the desired state. This action will then be saturated and sent 
        # automatically to Yumi.
        vel_r, vel_l = self.control_law.compute_target_state()
        
        # This is an action. For more information on the fields, please read the 
        # documentation of `self._solve_action()`.
        action = MixedVelocityYumiAction()
        action.control_space(MixedVelocityYumiAction.ControlSpace.INDIVIDUAL)
        action.velocity_right(vel_r)
        action.velocity_left(vel_l)
                
        return action


def main():
    # Set this Python script as a node in the "ROS network", to enable communication
    # with other nodes (i.e. the robot)
    rospy.init_node("dummy_controller", anonymous=False)
    
    # Instanciate the robot (this essentially opens a socket with the robot)
    yumi = YumiDevice()
    # Instantiate the controller
    controller = YumiDummyController(yumi)
    
    def shutdown_callback():
        print("Controller shutting down")
        # This function stops the internal loop of the controller, and sends 
        # some zero-velocity messages to stop Yumi.
        controller.stop()
    
    # Do not forget to stop the controller when killing ROS
    rospy.on_shutdown(shutdown_callback)
    
    # This method sets the controller as ready to send the policy. If this 
    # function doesn't get called, the controller simply sends the default 
    # policy in `self.default_policy()`.
    controller.ready()
    # Start the control loop. This is a blocking call and can be interrupted
    # only by calling `self.stop()` either in another thread or in the inner 
    # logic of the controller. This function is a wrapper for essentially 
    # these steps:
    #   1. fetch teh status of the robot
    #   2.1. compute the policy if both the robot and the controller are ready
    #   2.2. compute the default policy if only the robot is ready
    #   2.3. do nothing if neither the controller nor the robot are ready
    #   3. solve the action with the selected inverse kinematics solver
    #   4. send the command to the robot
    controller.start()


if __name__ == "__main__":
    main()
