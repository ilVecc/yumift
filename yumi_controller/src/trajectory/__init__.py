from .base import Param, Trajectory, MultiParam, MultiTrajectory, FakeTrajectory, sampled
from .base_impl import JointParam, PositionParam, QuaternionParam, PoseParam

from .polynomial import (
    CubicTrajectory, CubicPosTrajectory, CubicQuatTrajectory, CubicPoseTrajectory, 
    CubicPath, CubicPosePath
)


import time

class timeit():
    
    def __init__(self) -> None:
        self.init : float
    
    def __enter__(self):
        self.init = time.time()
        return self
    
    def __exit__(self, *args):
        elap = time.time() - self.init
        print(1/elap)