from setuptools import setup, find_packages

setup(
    name="trajectory",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        # 'console_scripts': [
        #     'my_tool = trajectory.my_tool:main'
        # ],
    },
    author="Sebastiano Fregnan",
    author_email="sebastiano@fregnan.me",
    description="Abstract trajectory objects and some useful implementations.",
    license="MIT",
    keywords=[],
    url="",
    classifiers=[]
)