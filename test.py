import numpy as np
import pandas as pd
from car_sharing_environment import CarsharingEnv

if __name__ == "__main__":
    nr_vehicles = 5
    env = CarsharingEnv()
    for iters in range(1):
        dayly_data = pd.read_csv("test_dataset.csv", sep=';')
        s = env.reset(dayly_data)
        done = False
        while not done:
            # always charge all vehicles
            act = np.ones(nr_vehicles)
            act = np.array([1, 1, 1, 0, 2])
            s, rew, dayly_data, done, _ = env.step(act, dayly_data)
            env.render(dayly_data)
            print("Reward in this step:", rew)
        env.episode_summary_statistics(dayly_data)


