# TODO make sense of this, i have no idea what i'm doing

from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=[
        "core_common",
        "core_controllers"
    ],
    package_dir={'': 'src'},
    requires=[]
)

setup(**d)