from typing import Any
from abc import ABCMeta, abstractmethod


class ControlLawError(Exception):
    pass

class AbstractControlLaw(object, metaclass=ABCMeta):
    """ Abstract interface for a generic control law.
        Computation flow consists of the following steps:
        1. (optional) `update_current_timestep()`, which changes the value 
            of the time interval between current state and desired state
        2. `update_current_state()`, which modifies the internal initial state
        3. `update_desired_state()`, which modifies the internal final state
        4. `compute_target_state()`, which computes the required action to 
            achive the desired movement
        
        These steps are also available using `update_and_compute()`, which 
        calls all the required methods in the correct order. A `clear()` method 
        is available to clear the internal variables of the class.
    """
    
    def __init__(self, initial_timestep: float = 0.):
        super().__init__()
        self.dt = initial_timestep
    
    @abstractmethod
    def clear(self):
        """ Reset the internal variables of the control law
        """
        raise NotImplementedError()
    
    def update_current_timestep(self, timestep: float):
        """ Update the time internal between current and desired state
        """
        self.dt = timestep
    
    @abstractmethod
    def update_current_state(self, state: Any):
        """ Update the "now" state of the controlled system
        """
        raise NotImplementedError()
    
    @abstractmethod
    def update_desired_state(self, desired: Any):
        """ Update the "next" state of the controlled system
        """
        raise NotImplementedError()
    
    @abstractmethod
    def compute_target_state(self) -> Any:
        """ Compute the required action that brings the system from the current
            state to the desired state. If no action is found, this method can
            raise a `ControlLawError` exception.
        """
        raise NotImplementedError()

    def update_and_compute(self, current_state: Any, desired_state: Any, timestep: float) -> Any:
        """ Utility method that updates timestep, current and desired state and 
            immediately computes the required action. Useful when no extra logic 
            is required between the updates and the computation.
        """
        self.update_current_timestep(timestep)
        self.update_current_state(current_state)
        self.update_desired_state(desired_state)
        return self.compute_target_state()
