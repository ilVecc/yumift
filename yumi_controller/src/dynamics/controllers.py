# TODO when swtiching to Python3.8, use @final on the methods
from abc import ABCMeta, abstractmethod #, final
from typing import Any, TypeVar, Type, Generic

import time
from threading import Lock


class AbstractDeviceState(object):
    
    def __init__(self):
        self.time : Any

TState = TypeVar("TState", bound=Type[AbstractDeviceState])
class AbstractDevice(Generic[TState], metaclass=ABCMeta):
    
    @abstractmethod
    def is_ready(self) -> bool:
        """ Returns the current readiness of the device.
        """
        raise NotImplementedError()

    @abstractmethod
    def read(self) -> TState:
        """ Read the current state of the device.
            This function must create a device state with updated time in a 
            thread-safe way.
        """
        raise NotImplementedError()

    @abstractmethod
    def send(self, command: Any):
        """ Send command to the device via the required hardware interface 
            (e.g. CANBUS, TCP socket). A good example might be preparing and 
            sending a velocity command message over ROS.
            
            :param command: a representation of the command the fits the 
                            requirements of the hardware interface
        """
        raise NotImplementedError()


class AbstractController(Generic[TState], metaclass=ABCMeta):
    """ Class for controlling a generic device, inherit this class and concretize
        every abstract function. `self.policy()` and `self.default_policy()` 
        output an action, which is then passed to `self._solve_action()`, which 
        finally produces a command sent by the device.
    """

    def __init__(self, device: AbstractDevice[TState]):
        # signal "controller is stopped"
        self._lock_controller_stop = Lock()
        self._controller_stop = False
        # signal "controller can command"
        self._lock_controller_ready = Lock()
        self._controller_ready = False
        # device handle
        self._device = device
        self._device_last_state: TState

    # execution functions

    def _inner_loop(self, rate):
        """ Internal blocking control loop. Overwrite this function to use 
            different rate-handling strategies. This function can be stopped 
            calling `self.stop()` from another thread or in the inner logic of 
            the controller.
        """
        dt = 1 / rate
        init = time.time()
        while not self.is_stopped():
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
        state = self._device_read()
        if self._device_is_ready():
            # run a computation step of the controller
            if self.is_ready():
                action = self._inner_policy(state)
            else:
                print("Controller not ready yet (sending default policy)")
                action = self.default_policy(state)
            # solve the action and send it
            command = self._solve_action(action)
            self._device_send(command)
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

    # status signals

    # TODO @final
    def is_stopped(self) -> bool:
        """ Returns the `self._controller_stop` flag
        """
        with self._lock_controller_stop:
            return self._controller_stop

    # TODO @final
    def is_ready(self) -> bool:
        """ Returns the `self._controller_ready` flag
        """
        with self._lock_controller_ready:
            return self._controller_ready

    # wrappers
    
    def _device_is_ready(self) -> bool:
        """ Logic for checking if the device is ready for I/O operations.
            This comprehends pre- and post- `self._device.is_ready()` logic.
            By default, simply call `self._device.is_ready()`.
        """
        return self._device.is_ready()
    
    def _device_read(self) -> TState:
        """ Read device state logic. This comprehends pre- and post- 
            `self._device.read()` logic. By default, simply call `self._device.read()`. 
            When overwriting this function, always return a state. 
        """
        self._device_last_state = self._device.read()
        return self._device_last_state

    def _device_send(self, command: Any):
        """ Send command to device logic. This comprehends pre- and post- 
            `self._device.send()` logic. By default, simply call `self._device.send()`. 
            When overwriting this function, return nothing. 
        """
        return self._device.send(command)

    def _inner_policy(self, state: TState) -> Any:
        """ Inner controller logic. This comprehends pre- and post- `self.policy()`
            logic. By default, simply call `self.policy()`. When overwriting this
            function, always return an action.
        """
        return self.policy(state)
    
    # controller-specific functions
    
    @abstractmethod
    def reset(self):
        """ Reset the interal logic of the controller.
            A good example might be setting the initial target to the current 
            state of the device, so the reading the device's status can be 
            quite useful here. Do not call this function during the 
            initialization of your controller.
        """
        raise NotImplementedError()

    @abstractmethod
    def default_policy(self, state: TState) -> Any:
        """ Default policy to compute when the controller is not ready yet.
            A good example might be a simple all-zeros command.
        """
        raise NotImplementedError()

    @abstractmethod
    def policy(self, state: TState) -> Any:
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
