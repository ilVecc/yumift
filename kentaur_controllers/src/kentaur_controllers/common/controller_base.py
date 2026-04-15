import numpy as np

from dynamicals.common.controllers import AbstractControllerAction

from yumift_controllers.common.controller_base import MixedVelocityYumiAction
from sleipner_controllers.common.controller_base import SE2TwistAction


class KentaurDeviceAction(AbstractControllerAction, dict):
    def __init__(self) -> None:
        super().__init__()
        self.action_yumi = MixedVelocityYumiAction()
        self.action_sleipner = SE2TwistAction(np.zeros(3))
