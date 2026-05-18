from typing import List

from threading import Lock

from yumift_controllers.common.device import YumiDualDeviceState, YumiDevice
from yumift_controllers.common.controller_base import YumiDualController, MixedVelocityYumiAction
from .routine_sm import RoutineStateMachine, Routine
from ..ik.solver import IKAlgorithm


class RoutinableYumiController(YumiDualController):

    def __init__(self, robot_handle: YumiDevice, iksolvers: List[IKAlgorithm], routines: List[Routine] = []):
        super().__init__(robot_handle, iksolvers)
        
        # routine variables
        self._lock_routine_request = Lock()
        self._routine_request = None
        self._routine_machine = RoutineStateMachine()
        for routine in routines:
            self._routine_machine.register(routine)
    
    def request_routine(self, name: str):
        """ Set the routine to run. This can be done either internally in
            the `self.policy()` function or externally in another thread. 
            If you do it internally, it will be executed in the next cycle.
        """
        with self._lock_routine_request:
            self._routine_request = name
    
    def _desired_policy(self, state: YumiDualDeviceState) -> MixedVelocityYumiAction:
        """ New internal policy for the controller. Now, before computing the 
            policy, run the requested rountine, if any is requested or already
            running. Otherwise, run the policy.
        """
        # copy the request to avoid locking it for long
        with self._lock_routine_request:
            request = self._routine_request
            self._routine_request = None
        
        # execute the request (if exists)
        action, done = self._routine_machine.run(state, request)
        if done is True:
            # routine just finished, reset controller first
            self.reset(state)
        elif action is not None:
            # action from routine exists, return it
            return action
        
        return self.policy(state)
    