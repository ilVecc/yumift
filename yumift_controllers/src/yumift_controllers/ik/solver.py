from abc import ABCMeta, abstractmethod
from typing import Dict, List, Union

import numpy as np

from dynamicals.common.robotics import RobotState


class IKAlgorithm(object, metaclass=ABCMeta):

    def __init__(self, name: str, can_init_late: bool = True):
        super().__init__()
        self.name = name
        self.can_init_late = can_init_late
        self._is_init = False

    def do_init(self) -> None:
        if not self._is_init:
            self.init()
            self._is_init = True

    @abstractmethod
    def init(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def solve(self, action: dict) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError()


class IKSolver(object):
    def __init__(self, algorithms: List[IKAlgorithm]):
        super().__init__()
        self._algorithms: Dict[str, IKAlgorithm] = dict()
        self._selected_algo : Union[str, None] = None
        for algo in algorithms:
            self.register(algo)

    def register(self, algorithm: IKAlgorithm, init_later: bool = False) -> bool:
        if algorithm.name in self._algorithms:
            print("Algorithm name already registered")
            return False
        self._algorithms[algorithm.name] = algorithm
        # init the algorithm if required
        if not self._algorithms[algorithm.name].can_init_late or not init_later:
            self._algorithms[algorithm.name].do_init()
        return True

    def switch(self, name: str) -> bool:
        if name not in self._algorithms:
            print("Algorithm name not registered, keeping previous algorithm")
            return False
        # stop current algorithm
        if self._selected_algo is not None:
            self._algorithms[self._selected_algo].stop()
        self._selected_algo = name
        # init the algorithm (this is idempotent if `init_later` was False when 
        # registering this algorithm)
        self._algorithms[self._selected_algo].do_init()
        return True

    def solve(self, action: dict, state: RobotState) -> np.ndarray:
        return self._algorithms[self._selected_algo].solve(action, state)

    def __call__(self, action: dict, state: RobotState):
        return self.solve(action, state)
