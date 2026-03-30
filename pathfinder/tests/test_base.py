import numpy as np
from .. import Param, MultiParam, Trajectory


class PointParam(Param):
    def __init__(self, coords: np.array) -> None:
        assert coords.size == 2
        super().__init__(coords)

class LinearTrajectory(Trajectory[PointParam]):
    def compute(self, t: float) -> PointParam:
        coords = self._param_init.value + (self._param_final.value - self._param_init.value) * t/self._duration
        return PointParam(coords)


def test_simple_trajectory():
    """ Showcase of how to use the base classes.
    """
    
    traj = LinearTrajectory()
    pi = PointParam(np.array([0, 0]))
    pf = PointParam(np.array([1,-1]))
    traj.update(pi, pf, 4)
    
    assert np.allclose(traj.compute(3).value, np.array([.75, -.75]))
    
    # how to create a MultiParam object
    path_param = MultiParam[PointParam](pi, 3)


if __name__ == "__main__":
    test_simple_trajectory()