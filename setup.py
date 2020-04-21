from setuptools import setup

setup(
    name="rl-interpretation-toolbox",
    version="0.0.1",
    author="pengzhenghao",
    author_email="pengzh@ie.cuhk.edu.hk",
    packages=['toolbox'],
    install_requires=[
        "gym==0.12.1",
        "lz4",
        "Markdown",
        "matplotlib",
        "opencv-python",
        "pytest==4.3.1",
        "tensorflow-probability",
        "yapf==0.27",
        "numpy==1.16.0",
        "gym[mujoco]",
        "gym[box2d]",
        "pandas",
        "yattag",
        "box2d-py==2.3.8",
        "sklearn",
        "seaborn",
        "distro",
        "gym[atari]",
        "gputil",
        "ray==0.8.4",
        "ray[tune]==0.8.4",
        "ray[rllib]==0.8.4",
        "ray[dashboard]==0.8.4",
        "ray[debug]==0.8.4",
        "gym-minigrid"
    ]
)
