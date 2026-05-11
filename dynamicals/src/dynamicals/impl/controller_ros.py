from abc import ABCMeta
from typing_extensions import override, Optional

from ..common.devices import TState, TCommand, AbstractDevice
from ..common.controllers import TAction, AbstractController

import rospy


class AbstractROSController(AbstractController[TState, TAction, TCommand], metaclass=ABCMeta):
    """ Abstract class for controlling a generic device over ROS.
        Inherit this class and concretize every abstract function. 
        `self.policy()` and `self.fallback()` output an action, which is then 
        passed to `self._solve_action()`, and finally produces the command sent 
        by the device.
    """

    def __init__(self, device : AbstractDevice[TState, TCommand]):
        super().__init__(device)
        self._default_command = device.__orig_bases__[0].__args__[1]()
        rospy.on_shutdown(self.stop)

    def dt(self, from_time : rospy.Time):
        return (rospy.Time.now() - from_time).to_sec()
        
    @override
    def spin(self, control_rate: float):
        """ ROS implementation of the original function.
        """
        rate = rospy.Rate(control_rate)
        while not rospy.is_shutdown() or not self.is_stopped():
            self.spin_once()
            rate.sleep()
    
    @override
    def start(self):
        rospy.loginfo("Controller will start up soon")
        super().start(250) # Hz
    
    @override
    def stop(self, stop_commands: int = 5):
        rospy.loginfo("Controller is stopping")
        super().stop()
        # send a stop command
        for i in range(stop_commands):
            self.device_send(self._default_command)
            rospy.loginfo(f"Sent default command ({i+1}/{stop_commands})")
    
    @override
    def fallback(self, state: TState) -> TCommand:
        return self._default_command
    