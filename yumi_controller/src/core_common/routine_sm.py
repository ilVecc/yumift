from abc import ABCMeta, abstractmethod
from typing import Dict, Tuple, Optional
from enum import Enum, auto

from dynamics.utils import RobotState


class Routine(object, metaclass=ABCMeta):

    def __init__(self, name: str, *args) -> None:
        self.name = name

    @abstractmethod
    def init(self, robot_state_init: RobotState) -> None:
        """ Initialize internal state of the routine.
        """
        raise NotImplementedError()

    @abstractmethod
    def action(self, robot_state_curr: RobotState) -> Tuple[dict, bool]:
        """ Compute the current command of the routine. Both an action and a "done"
            flag must be returned. Whenever `done == True`, the linked action is 
            discarded and instead `self.finish()` is run.
        """
        raise NotImplementedError()

    @abstractmethod
    def finish(self, robot_state_final: RobotState) -> None:
        """ Conclude the routine. This might be useful to reset internal variables.
        """
        raise NotImplementedError()


class RoutineStateMachine(object):

    class State(Enum):
        IDLING = auto()
        RUNNING = auto()
        COMPLETED = auto()

    def __init__(self):
        self._routines: Dict[str, Routine] = {}
        # state
        self._state = RoutineStateMachine.State.IDLING
        self._running = None

    def state(self):
        return self._state

    def register(self, routine: Routine) -> None:
        if routine.name in self._routines:
            print("Routine name already registered")
            return
        self._routines[routine.name] = routine

    def reset(self) -> None:
        self._state = RoutineStateMachine.State.IDLING
        self._running = None

    def run(self, robot_state: RobotState, name: Optional[str] = None):

        if name is not None:
            if self.state() == RoutineStateMachine.State.RUNNING:
                # another routine is running!
                print("Cannot run a routine when another one is already running!")
            elif self.state() == RoutineStateMachine.State.IDLING:
                # set routine and start running
                try:
                    self._running = self._routines[name]
                    print(f"Running routine \"{name}\"")
                except Exception:
                    print(f"Routine \"{name}\" not registered! Resuming execution")
            else:
                # i cannot be here
                pass

        done = False

        if self._state == RoutineStateMachine.State.IDLING and self._running == None:
            # routine state machine is idling and nothing is requested (avoid extra ifs)
            return None, done

        if self._state == RoutineStateMachine.State.IDLING and self._running != None:
            # we enter here only when a routine is requested, and we init it.
            # no action is created here because we immediately enter the following if
            self._state = RoutineStateMachine.State.RUNNING
            self._running.init(robot_state)

        if self._state == RoutineStateMachine.State.RUNNING:
            # generate the action and check if we are done. if so, set the state
            # and continue in the next if block
            action, done = self._running.action(robot_state)
            if done:
                self._state = RoutineStateMachine.State.COMPLETED

        if self._state == RoutineStateMachine.State.COMPLETED:
            # no routine is running anymore, remove it.
            # no action is created because we enter here only when the previous
            # routine `.action()` call produced `done == True`, thus we already 
            # have an action
            print("Routine done")
            self._state = RoutineStateMachine.State.IDLING
            self._running.finish(robot_state)
            self._running = None

        return action, done
