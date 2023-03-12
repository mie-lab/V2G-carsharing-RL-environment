import numpy as np
from Experiments.environment import CarsharingEnv

if __name__ == "__main__":
    nr_vehicles = 5
    env = CarsharingEnv(nr_vehicles=nr_vehicles)
    for iters in range(1):
        s = env.reset()
        done = False
        while not done:
            # always charge all vehicles
            act = np.ones(nr_vehicles)
            s, rew, done, _ = env.step(act)
            env.render()
            # print("Reward in this step:", rew)
