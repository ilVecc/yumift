import rospkg
import yaml
from pathlib import Path


def load_config(filename : str):
    pkg_path = Path(rospkg.RosPack().get_path("kentaur_controllers")) / "config" / filename
    with open(str(pkg_path)) as f:
        data = yaml.safe_load(f)
    return data
