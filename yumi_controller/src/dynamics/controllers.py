# TODO when swtiching to Python3.8, use @final on the methods
from abc import ABCMeta, abstractmethod #, final
from typing import Any

import time
from threading import Lock


class AbstractController(object, metaclass=ABCMeta):
    """ Class for controlling a generic device, inherit this class and concretize
        every abstract function. `self.policy()` and `self.default_policy()` 
        output an action, which is then passed to `self._solve_action()`, which 
        finally produces a command sent using `self._send_command()`.
    """

    def __init__(self):
        # signal "controller is stopped"
        self._lock_controller_stop = Lock()
        self._controller_stop = False
        # signal "controller can command"
        self._lock_controller_ready = Lock()
        self._controller_ready = False
        # signal "device can operate"
        self._lock_device_ready = Lock()
        self._device_ready = False
        # status (together with `self._device_ready`) of the system
        self._device_state: Any
        self._device_time: Any

    # execution functions

    def _inner_loop(self, rate):
        """ Internal blocking control loop. Overwrite this function to use 
            different rate-handling strategies. This function can be stopped 
            calling `self.stop()` from another thread or in the inner logic of 
            the controller.
        """
        dt = 1 / rate
        init = time.time()
        while not self.is_controller_stopped():
            self.cycle()
            time.sleep(max(0, dt - (time.time() - init)))
            init = time.time()

    # TODO @final
    def cycle(self):
        """ Compute and send commands to the device. In order, this function:
            1. fetches and updates current device status
            2. if device is not ready, does nothing
            3. else, if controller is ready, calculates the action using the 
                policy, otherwise using the default policy
            4. transforms the action to a command
            5. sends the command
        """
        # fetch state of device
        self.fetch_device_status()
        if self.is_device_ready():
            # run a computation step of the controller
            if self.is_controller_ready():
                action = self._inner_policy()
            else:
                print("Controller not ready yet (sending default policy)")
                action = self.default_policy()
            # solve the action and send it
            command = self._solve_action(action)
            self._send_command(command)
        else:
            print("Device not ready yet (sending nothing)")

    # signal status

    # TODO @final
    def start(self, rate: float):
        """ Control the device at a given rate. This function is blocking and 
            uses `self._inner_loop()` for the rate regulation.
            To stop this function, call `self.stop()` from an other thread or 
            via inner controller logic. 
            
            :param rate: control rate of the controller [Hz]
        """
        with self._lock_controller_stop:
            self._controller_stop = False
        self._inner_loop(rate)
    
    # TODO @final
    def stop(self):
        """ Stop completely the controller. The controller can still be used 
            directly via `self.cycle()`.
            This operation simply stops the blocking `self.start()`.
        """
        with self._lock_controller_stop:
            self._controller_stop = True
    
    # TODO @final
    def ready(self):
        """ Enable the controller to send policy commands. 
            Call this function when the inner logic of the controller is ready
            to send intended commands.
            Use `self.pause()` to undo this operation.
        """
        with self._lock_controller_ready:
            self._controller_ready = True

    # TODO @final
    def pause(self):
        """ Stop sending policy commands and start sending default policy commands.
            Use `self.ready()` to undo this operation.
        """
        with self._lock_controller_ready:
            self._controller_ready = False

    # TODO @final
    def _set_device_ready(self, flag) -> bool:
        """ Sets the `self._device_ready` flag
        """
        with self._lock_device_ready:
            self._device_ready = flag
    
    # status signals

    # TODO @final
    def is_controller_stopped(self) -> bool:
        """ Returns the `self._controller_stop` flag
        """
        with self._lock_controller_stop:
            return self._controller_stop

    # TODO @final
    def is_controller_ready(self) -> bool:
        """ Returns the `self._controller_ready` flag
        """
        with self._lock_controller_ready:
            return self._controller_ready

    # TODO @final
    def is_device_ready(self) -> bool:
        """ Returns the `self._device_ready` flag
        """
        with self._lock_device_ready:
            return self._device_ready
    
    # controller-specific functions

    @abstractmethod
    def fetch_device_status(self):
        """ Read the current state of the device and update its status.
            ATTENTION: this function MUST update `self._device_state` and `self._device_time` 
            in a thread-safe way. Also, it MUST use `self._set_device_ready()` to update 
            the current status of the device (e.g. ready, waiting, error). 
        """
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        """ Reset the interal logic of the controller.
            A good example might be setting the initial target to the current 
            state of the device, so the function `self.fetch_device_status()`
            can be quite useful here. Do not call this function during the 
            initialization of your controller.
        """
        raise NotImplementedError()

    @abstractmethod
    def default_policy(self) -> Any:
        """ Default policy to compute when the controller is not ready yet.
            A good example might be a simple all-zeros command.
        """
        raise NotImplementedError()

    def _inner_policy(self) -> Any:
        """ Inner controller logic. This comprehends pre- and post- `self.policy()`
            logic. By default, simply call `self.policy()`. When overwriting this
            function, always return an action.
        """
        return self.policy()
    
    @abstractmethod
    def policy(self) -> Any:
        """ Policy to compute when the controller is ready.
        """
        raise NotImplementedError()

    @abstractmethod
    def _solve_action(self, action: Any) -> Any:
        """ Solve the internal representation of the action to a more concrete 
            command. A good example might be a robot cartesian action to be solved
            via inverse kinematics to a joint velocity command.
        """
        raise NotImplementedError()

    @abstractmethod
    def _send_command(self, command: Any):
        """ Send command to the device via the required hardware interface 
            (e.g. CANBUS, TCP socket). A good example might be preparing and 
            sending a velocity command message over ROS.
            
            :param command: a representation of the command the fits the 
                            requirements of the hardware interface
        """
        raise NotImplementedError()
