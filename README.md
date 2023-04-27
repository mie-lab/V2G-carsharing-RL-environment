# Reinforcement Learning-Enabled Simulation Environment for Optimizing Vehicle-to-Grid Charging Strategies in Car-Sharing Systems
This repository contains an open-source simulation environment designed for testing and optimizing vehicle-to-grid (V2G) charging strategies in car sharing systems. The environment supports a range of reinforcement learning algorithms and can be used to test other approaches and strategies as well. The environment includes a tuple observation space and a discrete action space, along with a reward function that represents a broad range of economic costs and revenues, including opportunity costs. The reward function encourages efficient charging and discharging of vehicles, while taking into account factors such as energy prices, penalties, and revenue from renting out vehicles.

## Installation:
This repo can then be installed in editable mode. In an activated virtual environment, cd into this folder and run
```
pip install -e .
```
This will execute the `setup.py` file and install required dependencies.
