import numpy as np

from dynamicals.common.controllers import AbstractControllerAction


class SE2TwistAction(AbstractControllerAction):
    
    def __init__(self, twist_SE2 : np.ndarray = np.zeros(3)) -> None:
        assert twist_SE2.shape == (3,)
        super().__init__()
        self.twist_SE2 = twist_SE2

    @staticmethod
    def from_twist_SE3(twist_SE3 : np.ndarray):
        return SE2TwistAction(np.array([twist_SE3[0], twist_SE3[1], twist_SE3[5]]))
    