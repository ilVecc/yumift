# TODO when swtiching to Python3.8, use @final on the methods
from abc import ABCMeta, abstractmethod
from typing import Any, TypeVar, Type, Generic, final

import time
from threading import Lock

from .devices import TState, TCommand, AbstractDevice


class AbstractControllerAction(object):
    def __init__(self):
        self.time : Any

TAction = TypeVar("TAction", bound=Type[AbstractControllerAction])


# TODO could be useful
class DeviceUnavailableException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class AbstractController(Generic[TState, TAction, TCommand], metaclass=ABCMeta):
    """ Class for controlling a generic device, inherit this class and concretize
        every abstract function. `self.policy()` and `self.fallback()` 
        output an action, which is then passed to `self._solve_action()`, which 
        finally produces a command sent by the device.
    """

    def __init__(self, device : AbstractDevice[TState, TCommand]):
        # signal "controller is stopped"
        self._lock_controller_stopped = Lock()
        self._controller_stopped = False
        # signal "controller can command"
        self._lock_controller_ready = Lock()
        self._controller_ready = False
        # device handle
        self._device = device
        self._device_last_state : TState

    # execution functions

    def spin(self, rate):
        """ Internal blocking control loop. Overwrite this function to use 
            different rate-handling strategies. This function can be stopped 
            calling `self.stop()` from another thread or in the default/desired 
            logic of the controller.
        """
        dt = 1 / rate
        init = time.time()
        while not self.is_stopped():
            self.spin_once()
            time.sleep(max(0, dt - (time.time() - init)))
            init = time.time()

    # TODO @final  maybe not
    def spin_once(self):
        """ Compute and send commands to the device. In order, this function:
            1. fetches and updates current device status
            2. if device is not ready, does nothing
            3. else, if controller is ready, calculates the action using the 
                policy, otherwise using the default policy
            4. transforms the action to a command
            5. sends the command
        """
        if self.device_is_ready():
            # fetch state of device
            state = self.device_read()
            # run a computation step of the controller
            if self.is_ready():
                action = self._desired_policy(state)
            else:
                action = self._default_policy(state)
            # solve the action and send it
            command = self.solve_action(state, action)
            self.device_send(command)
        else:
            # TODO this is both time-expensive and undesired, find a different logging system
            print("Device not ready yet (idling)")

    # signal status

    # TODO @final
    def start(self, rate: float):
        """ Control the device at a given rate. This function is blocking and 
            uses `self._desired_loop()` for the rate regulation.
            To stop this function, call `self.stop()` from an other thread or 
            via default/desired controller logic. When the controller is started, 
            no effort is put into checking the readyness of the controll, as that
            will be taken care by the desired loop; essentially, when the
            controller is started, either the default or the desired policy 
            are immediately executed at given rate. 
            
            :param rate: control rate of the controller [Hz]
        """
        with self._lock_controller_stopped:
            self._controller_stopped = False
        self.spin(rate)
    
    # TODO @final
    def stop(self):
        """ Stop completely the controller. The controller can still be used 
            directly via `self.cycle()`.
            This operation simply stops the blocking `self.start()`.
        """
        with self._lock_controller_stopped:
            self._controller_stopped = True
    
    # TODO @final
    def ready(self):
        """ Enable the controller to send policy commands. 
            Call this function when the desired logic of the controller is ready
            to send desired commands. This function must be called in order to
            switch from default to desired policy.
            Use `self.pause()` to undo this operation.
        """
        with self._lock_controller_ready:
            self._controller_ready = True

    # TODO @final
    def pause(self):
        """ Stop sending desired policy commands and start sending default 
            policy commands.
            Call this function when the desired logic of the controller cannot
            be used anymore. This function must be called in order to switch 
            from desired to default policy.
            Use `self.ready()` to undo this operation.
        """
        with self._lock_controller_ready:
            self._controller_ready = False

    # status signals

    # TODO @final
    def is_stopped(self) -> bool:
        """ Returns the `self._controller_stop` flag
        """
        with self._lock_controller_stopped:
            return self._controller_stopped

    # TODO @final
    def is_ready(self) -> bool:
        """ Returns the `self._controller_ready` flag
        """
        with self._lock_controller_ready:
            return self._controller_ready

    # wrappers for pre- and post- conditions
    
    def device_reset(self) -> bool:
        """ Logic for resetting the device.
            This comprehends pre- and post- `self._device.reset()` logic.
            By default, this simply calls `self._device.reset()`.
            When overwriting this method, keep the same signature.
        """
        return self._device.reset()
    
    def device_is_ready(self) -> bool:
        """ Logic for checking if the device is ready for I/O operations.
            This comprehends pre- and post- `self._device.is_ready()` logic.
            By default, this simply calls `self._device.is_ready()`.
            When overwriting this method, keep the same signature.
        """
        return self._device.is_ready()
    
    def device_read(self) -> TState:
        """ Logic for reading device state.
            This comprehends pre- and post- `self._device.read()` logic. 
            By default, this simply calls `self._device.read()`, storing the result.
            When overwriting this method, keep the same signature.
        """
        self._device_last_state = self._device.read()
        return self._device_last_state

    def device_send(self, command: TCommand) -> None:
        """ Logic for sending command to the device. 
            This comprehends pre- and post- `self._device.send()` logic. 
            By default, this simply calls `self._device.send()`. 
            When overwriting this method, keep the same signature.
        """
        return self._device.send(command)

    def _default_policy(self, state: TState) -> TAction:
        """ Default controller logic. 
            This comprehends pre- and post- `self.fallback()` logic. 
            By default, this simply calls `self.fallback()`. 
            When overwriting this method, keep the same signature.
        """
        # TODO this is both time-expensive and undesired, find a different logging system
        print("Controller not ready yet (fallback)")
        return self.fallback(state)
    
    def _desired_policy(self, state: TState) -> TAction:
        """ Desired controller logic. 
            This comprehends pre- and post- `self.policy()` logic. 
            By default, this simply calls `self.policy()`. 
            When overwriting this method, keep the same signature.
        """
        return self.policy(state)
    
    # controller-specific functions
    
    @abstractmethod
    def reset(self, state: TState):
        """ Reset the interal logic of the controller.
            A good example might be setting the initial target to the current 
            state of the device, so the reading the device's status can be 
            quite useful here. Do not call this function during the 
            initialization of your controller.
        """
        raise NotImplementedError()

    @abstractmethod
    def fallback(self, state: TState) -> TAction:
        """ Fallback policy to compute when the controller is not ready yet,
            e.g. "do nothing" action.
        """
        raise NotImplementedError()

    @abstractmethod
    def policy(self, state: TState) -> TAction:
        """ Desired policy to compute when the controller is ready.
        """
        raise NotImplementedError()

    @abstractmethod
    def solve_action(self, state: TState, action: TAction) -> TCommand:
        """ Solve the internal representation of the action to a more command,
            e.g. solve a 6D cartesian action via inverse kinematics and find 
            joint velocity command with a particular solver.
            
            :param action: the action to be solved
            :returns: the command obtained from the action
        """
        raise NotImplementedError()
