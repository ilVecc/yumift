import rospkg
import yaml
from pathlib import Path

from dynamicals.utils import Frame
from pathfinder.base_impl import PoseParam


def load_config(filename : str):
    pkg_path = Path(rospkg.RosPack().get_path("kentaur_controllers")) / "config" / filename
    with open(str(pkg_path)) as f:
        data = yaml.safe_load(f)
    return data

def PoseParam_to_Frame(pose_param: PoseParam):
    return Frame(pose_param.pos, pose_param.rot, pose_param.vel)
