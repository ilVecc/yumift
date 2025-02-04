from setuptools import setup, find_packages

setup(
    name="dynamics",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        # 'console_scripts': [
        #     'my_tool = dynamics.my_tool:main'
        # ],
    },
    author="Sebastiano Fregnan",
    author_email="sebastiano@fregnan.me",
    description="Dynamical models and control laws for robotic and 6D spaces.",
    license="MIT",
    keywords=[],
    url="",
    classifiers=[]
)