from setuptools import setup, find_packages

setup(
    name="dynamicals",
    version="0.1.0",
    description="Dynamical models and control laws for robotic and 6D spaces.",
    author="Sebastiano Fregnan",
    author_email="sebastiano@fregnan.me",
    url="",
    license="MIT",
    keywords=[],
    classifiers=[],
    #
    package_dir={"": "src"},
    packages=find_packages(where="src/"),
    entry_points={},
    python_requires=">=3.8, <4",
    install_requires=["numpy", "numpy-quaternion", "quadprog"]
)