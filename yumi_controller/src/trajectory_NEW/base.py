from typing import Any, List, Generic, Type, TypeVar, Union
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

TParam = TypeVar("TParam", bound=Type[Param])
class Trajectory(Generic[TParam], metaclass=ABCMeta):
    """ Class representing a generic trajectory filling the gaps between two points.
    """
    def __init__(self, param_init: TParam, param_final: TParam) -> None:
        """ Initialize the trajectory.
            :param param_init: initial parameter of the trajectory
            :param param_final: final parameter of the trajectory
        """
        self._param_init = param_init
        self._param_final = param_final
    
    @abstractmethod
    def compute(self, t: Any) -> TParam:
        """ Calculate the next target of the trajectory.
        """
        raise NotImplementedError("A target calculation method must be specified")











class FakeTrajectory(Trajectory[Any]):
    """ Trajectory useful when constructing a complex path which does not use
        an underlying Trajectory object (i.e. in a double path, two objects
        are needed instead of one). Remember to avoid using methods `.compute()` 
        and `.update()` if not necessary. 
    """
    def __init__(self) -> None:
        super().__init__(None, None)
    
    def compute(self, t) -> Any:
        return None
    

class ConstantTrajectory(Trajectory[Any]):
    """ Trajectory useful when constructing a complex path which remains constant
        for some time. Remember to avoid using methods `.compute()` and `.update()` 
        if not necessary. 
    """
    def __init__(self, const_param: TParam) -> None:
        super().__init__(const_param, const_param)
    
    def compute(self, t) -> Any:
        return self._param_init






















### MULTI-TRAJECTORY PLAN

# TODO make this a Param again, and find a way to get inherit from TParam
class PathParam(Generic[TParam]):
    """ Class for storing points for a multi-point trajectory
    """
    def __init__(self, param: TParam, time_instant: float = 0., time_duration: float = None):
        """ Convert a Parameter into a Path parameter providing either a time 
            instant (i.e. when to reach this parameter) or a time duration (i.e. 
            how long to reach this parameter). Provide either the instant or the 
            duration, and try to be consistent with the choice throughout the path.
            In case both are provided (i.e. `time_instant > 0` and `time_duration 
            is not None`), a list of resolution criteria is used when creating 
            the trajectory (see `Path` object).
            :param param: the trajectory parameter to reach
            :param time_instant: when to reach `param`
            :param time_duration: how long to reach `param`
        """
        super().__init__()
        self.param = param
        self.duration = time_duration
        self.instant = time_instant
    
    @classmethod
    def with_duration(cls, duration: float = 0., *args, **kwargs) -> "PathParam[TParam]":
        param = TParam(*args, **kwargs)
        return cls(param, 0, duration)
    
    @classmethod
    def with_instant(cls, instant: float = 0., *args, **kwargs) -> "PathParam[TParam]":
        param = TParam(*args, **kwargs)
        return cls(param, instant)


# TODO implement this dream of a description
class Path(Generic[TParam]):
    """ Generates a path passing through a list of points using the provided trajectories.
    
        The path can be constructed in "single trajectory" mode, specifying just 
        one Trajectory (first element), or "multi trajectory" mode, alternating 
        Trajectory and PathParam objects (with two consecutive PathParam objects, 
        reuse the last trajectory; with town consecutive Trajectory objects, raise
        exception). Time instants are checked and possibly calculated when just 
        durations are provided. 
        
        When a parameter provides both an instant and a duration, its timing is 
        computed as follows (generally, instants are prioritized over durations):
        0) for the first parameter, recompute the duration from the instant;
        1) if `time_instant - time_duration` is after the previous parameter's
           instant, keep both (the trajectory will start delayed wrt the 
           previous parameter);
        2) otherwise, recompute the duration based on the previous parameter's instant
        
        :param path_description: either a `[Param, Trajectory, Param, ...]` sequence
                                 or a `[Trajectory, Param, Param, ...]` sequence
    """
    def __init__(self, *path_description: Union[PathParam[TParam], Trajectory[TParam]]) -> None:
        
        assert len(path_description) >= 3, "The description must contain at least two points and a trajectory"
        
        trajectories = []  # len(.) = n-1
        parameters = []    # len(.) = n
        
        # if first element is a Trajectory, expect all the others to be Param
        if isinstance(path_description[0], Trajectory[TParam]):
            
            # filter the remaining
            params = path_description[1:]
            
            # deal with the first parameter
            assert isinstance(params[0], PathParam[TParam]), "In \"single trajectory\" description, all elements after the first one must be PathParam"
            if params[0].instant != 0 or params[0].duration != None:
                raise AssertionError("The first parameter must not specify time information (use default)")
            params[0].duration = 0
            
            parameters.append(params[0])
            
            # deal with all the others
            for i in range(len(params[1:])):
                assert isinstance(params[i], PathParam[TParam]), "In \"single trajectory\" description, all elements after the first one must be PathParam"
                self._sanitize_param(params[i-1], params[i])
                
                trajectories.append(path_description[0])  # TODO this is not efficient at all
                parameters.append(params[i])
            
        else:
            
            params = path_description
            
            # deal with the first parameter
            assert isinstance(params[0], PathParam[TParam]), "In \"multi trajectory\" description, the first element must be a PathParam"
            if params[0].instant != 0 or params[0].duration != None:
                raise AssertionError("The first parameter must not specify time information (use default)")
            params[0].duration = 0
            
            parameters.append(params[0])
            
            
            # deal with the first trajectory
            assert isinstance(params[0], PathParam[TParam]), "In \"multi trajectory\" description, the second element must be a Trajectory"
            if params[0].instant != 0 or params[0].duration != None:
                raise AssertionError("The first parameter must not specify time information (use default)")
            params[0].duration = 0
            
            parameters.append(params[0])
            
            
            # deal with all the others
            for i in range(len(params[1:])):
                if isinstance(params[i], PathParam[TParam]):
                    self._sanitize_param(params[i-1], params[i])
                
                    trajectories.append(path_description[0])  # TODO this is not efficient at all
                    parameters.append(params[i])
                elif isinstance(params[i], Trajectory[TParam]):
                    
                else:
                    raise AssertionError("In \"single trajectory\" description, all elements after the first one must be PathParam")
            
            
            
            
        
            
            
            
            
        
        
        
        self._traj_type = trajectory
        self._path_params: List[PathParam[TParam]] = []
        self._timemarks: np.ndarray
        self._segment_new: bool
        self._segment_idx_prev: int
        self._segment_idx_curr: int
        assert path_parameters[0].duration == 0, "First path parameter must have no duration"
        timemarks = np.cumsum([p.duration for p in path_parameters])
        super().update(path_parameters[0].param, path_parameters[-1].param, timemarks[-1])
        self._path_params = path_parameters
        self._timemarks = timemarks
        self._segment_new = True
        self._segment_idx_prev = -1
        self._segment_idx_curr = -1
        super().__init__()
    
    
    def _sanitize_param(self, prev: PathParam, curr: PathParam):
        if curr.instant != 0:
            if curr.duration is not None and prev.instant < (curr.instant - curr.duration):
                # TODO add a ConstantTrajectory here between the to params
                #      otherwise, keep last trajectory last output
                pass
            else:
                curr.duration = curr.instant - prev.instant
            
        elif curr.duration > 0:
            curr.instant = prev.instant + curr.duration
            
        else:
            raise AssertionError(f"No time information for parameter {curr}")
        
        return curr
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def clear(self) -> None:
        self._path_params = []
        self._timemarks = np.array([])
        self._segment_new = True
        self._segment_idx_prev = -1
        self._segment_idx_curr = -1
        super().clear()
    
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

PlanOrTrajType = TypeVar("PlanOrTrajType", Type[Path], Type[Trajectory])
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
    
    path_param = PathParam[PointParam](pi, 3)
    