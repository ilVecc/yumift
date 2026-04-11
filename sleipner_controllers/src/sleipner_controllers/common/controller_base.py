import numpy as np

from dynamicals.common.controllers import AbstractControllerAction


class SE2TwistAction(AbstractControllerAction):
    
    def __init__(self, twist_SE2: np.ndarray) -> None:
        assert twist_SE2.shape == (3,)
        super().__init__()
        self.twist_SE2 = twist_SE2
