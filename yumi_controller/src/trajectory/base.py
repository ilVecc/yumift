from typing import Any, List, Generic, Type, TypeVar
from functools import wraps
from abc import abstractmethod, ABCMeta
import numpy as np


################################################################################
##                                    BASE                                    ##
################################################################################

### TRAJECTORY

class Param(object):
    def __init__(self, *fields: Any) -> None:
        self._fields = list(fields)
    
    @property
    def value(self):
        return self.deriv(0)
    
    @property
    def speed(self):
        return self.deriv(1)
    
    @property
    def curve(self):
        return self.deriv(2)
    
    def deriv(self, order: int):
        if order < len(self._fields):
            return self._fields[order]
        return None

# TODO fix time between 0 and 1 and add a time-scaling/-shifting utility
TParam = TypeVar("TParam", bound=Type[Param])
class Trajectory(Generic[TParam], metaclass=ABCMeta):
    """ Class representing a generic trajectory filling the gaps between two points.
    """
    def __init__(self) -> None:
        """ Initialize the trajectory.
            When subclassing, always call this method at the end of the subclassed .__init__()
        """
        self._param_init: TParam
        self._param_final: TParam
        self._duration: float
        self.clear()
    
    def clear(self) -> None:
        """ Reset the trajectory. 
            When subclassing, always call this method at the end of the subclassed .clear()
        """
        self._param_init = None
        self._param_final = None
        self._duration = 0
    
    def update(self, param_init: TParam, param_final: TParam, duration: float) -> None:
        """ Set params of the trajectory. For safety, .clear() is automatically called before setting anything. 
            When subclassing, always call this method at the beginning of the subclassed .update()
        """
        self.clear()
        self._param_init = param_init
        self._param_final = param_final
        self._duration = duration
    
    @abstractmethod
    def compute(self, t: float) -> TParam:
        """ Calculate the next target of the trajectory.
        """
        raise NotImplementedError("A target calculation method must be specified")


### MULTI-TRAJECTORY PLAN

# TODO make this a Param again, and find a way to get inherit from TParam
class MultiParam(Generic[TParam]):
    """ Class for storing points for a multi-point trajectory
    """
    def __init__(self, param: TParam, duration: float = 0.):
        super().__init__()
        self.param = param
        self.duration = duration
    
    @classmethod
    def make_param(cls, duration: float = 0., *args, **kwargs) -> "MultiParam[TParam]":
        param = TParam(*args, **kwargs)
        return cls(param, duration)

class MultiTrajectory(Trajectory[TParam]):
    """ Generates a trajectory from a list of points.
    """
    def __init__(self, trajectory: Trajectory[TParam]) -> None:
        self._traj_type = trajectory
        self._path_params: List[MultiParam[TParam]] = []
        self._timemarks: np.ndarray
        self._segment_new: bool
        self._segment_idx_prev: int
        self._segment_idx_curr: int
        super().__init__()
    
    def clear(self) -> None:
        self._path_params = []
        self._timemarks = np.array([])
        self._segment_new = True
        self._segment_idx_prev = -1
        self._segment_idx_curr = -1
        super().clear()
    
    def update(self, path_parameters: List[MultiParam[TParam]]) -> None:
        assert path_parameters[0].duration == 0, "First path parameter must have no duration"
        timemarks = np.cumsum([p.duration for p in path_parameters])
        super().update(path_parameters[0].param, path_parameters[-1].param, timemarks[-1])
        self._path_params = path_parameters
        self._timemarks = timemarks
        self._segment_new = True
        self._segment_idx_prev = -1
        self._segment_idx_curr = -1
    
    def _update_segment(self, t) -> int:
        """ Updates the current target trajectory parameters or which is the 
            current segment on the trajectory.
        """
        self._segment_idx_curr: int = np.searchsorted(self._timemarks[:-1], t, side="right")
        self._segment_new = self._segment_idx_curr != self._segment_idx_prev
        if self._segment_new:
            param_init = self._path_params[self._segment_idx_curr-1]
            param_final = self._path_params[self._segment_idx_curr]
            self._traj_type.update(param_init.param, param_final.param, param_final.duration)
            self._segment_idx_prev = self._segment_idx_curr
        return t - self._timemarks[self._segment_idx_curr-1]  # time in segment period

    def is_new_segment(self) -> bool:
        """ Returns True if a new segment has been entered, only shows true
            once for each segment. Used to check if new gripper commands should be sent.
        """
        return self._segment_new
    
    def get_current_segment(self) -> int:
        """ Returns the index of the initial point of the current trajectory.
        """
        return self._segment_idx_curr

    def compute(self, t: float) -> TParam:
        """ Calculates the next target using the current segment. """
        t = max(0, t)
        t = self._update_segment(t)
        return self._traj_type.compute(t)

### MISC

PlanOrTrajType = TypeVar("PlanOrTrajType", Type[MultiTrajectory], Type[Trajectory])
def sampled(trajectory_class: PlanOrTrajType) -> PlanOrTrajType:
    func_init = trajectory_class.__init__
    func_clear = trajectory_class.clear
    func_update = trajectory_class.update
    func_compute = trajectory_class.compute
    
    @wraps(func_init)
    def __init__(self: PlanOrTrajType, dt, *args, **kwargs):
        self._dt = dt
        self._t = 0
        func_init(self, *args, **kwargs)
    
    @wraps(func_clear)
    def clear(self: PlanOrTrajType, *args):
        self._t = 0
        func_clear(self, *args)
    
    @wraps(func_update)
    def update(self: PlanOrTrajType, *args):
        self._t = 0
        func_update(self, *args)
    
    @wraps(func_compute)
    def compute(self: PlanOrTrajType):
        self._t += self._dt
        t = self._t
        return func_compute(self, t)

    trajectory_class.__init__ = __init__
    trajectory_class.clear = clear
    trajectory_class.update = update
    trajectory_class.compute = compute
    return trajectory_class


class FakeTrajectory(Trajectory[Any]):
    """ Trajectory useful when constructing a complex path which does not use
        an underlying Trajectory object (i.e. in a double path, two objects
        are needed instead of one). Remember to avoid using methods .compute() 
        and .update() if not necessary. """
    def __init__(self) -> None:
        super().__init__()
    def compute(self, t) -> Any:
        return None


if __name__ == "__main__":
    class PointParam(Param):
        def __init__(self, coords: np.array) -> None:
            assert coords.size == 2
            super().__init__(coords)
    
    class LinearTrajectory(Trajectory[PointParam]):
        def compute(self, t: float) -> PointParam:
            coords = self._param_init.value + (self._param_final.value - self._param_init.value) * t/self._duration
            return PointParam(coords)
    
    traj = LinearTrajectory()
    pi = PointParam(np.array([0, 0]))
    pf = PointParam(np.array([1,-1]))
    traj.update(pi, pf, 4)
    
    assert np.allclose(traj.compute(3).value, np.array([.75, -.75]))
    
    path_param = MultiParam[PointParam](pi, 3)
    