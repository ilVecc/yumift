from abc import ABCMeta, abstractmethod
from typing import Any, TypeVar, Type, Generic


class AbstractDeviceState(object):
    def __init__(self):
        self.time : Any

class AbstractDeviceCommand(object):
    def __init__(self):
        self.time : Any

TState = TypeVar("TState", bound=Type[AbstractDeviceState])  # TODO need covariant=True ?
TCommand = TypeVar("TCommand", bound=Type[AbstractDeviceCommand])

class AbstractDevice(Generic[TState, TCommand], metaclass=ABCMeta):
    
    @abstractmethod
    def reset(self) -> bool:
        """ Run the reset checklist of the device.
            This function could be useful when a device needs a specific reset 
            procedure, e.g. after a readiness status change.
        
            :returns: a flag that describes the completion of the reset checklist 
        """
        raise NotImplementedError()
    
    @abstractmethod
    def is_ready(self) -> bool:
        """ Returns the current readiness of the device.
        
            :returns: a flag that describes the "ready" state of the device
        """
        raise NotImplementedError()

    @abstractmethod
    def read(self) -> TState:
        """ Read the current state of the device.
            This function must create a device state with updated time in a 
            thread-safe way.
            
            :returns: a representation of the state that fits the requirements
                      of the hardware interface
        """
        raise NotImplementedError()

    @abstractmethod
    def send(self, command: TCommand):
        """ Send command to the device via the required hardware interface 
            (e.g. CANBUS, TCP socket). A good example might be preparing and 
            sending a velocity command message over ROS.
            
            :param command: a representation of the command that fits the 
                            requirements of the hardware interface
        """
        raise NotImplementedError()

TDevice = TypeVar("TDevice", bound=Type[AbstractDevice])
