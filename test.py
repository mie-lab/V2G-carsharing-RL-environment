import numpy as np
import pandas as pd
from car_sharing_environment import CarsharingEnv
import pandas as pd
import math
from sqlalchemy import create_engine

if __name__ == "__main__":

    # Create a database connection string
	engine = create_engine('postgresql://username:password@host:port/database_name')

    # simulate sinlge day
    nr_iterations = 1 # number of days
    env = CarsharingEnv(nr_vehicles=4233)
    if nr_iterations > 577:
        nr_iterations = 577
    count = 0
    for week_nr in range(0, math.ceil(nr_iterations / 7)):
        sql = "SELECT * FROM msc_2023_dominik.discrete_weeks_{}".format(week_nr)
        data = pd.read_sql(sql, engine)
        vehicle_list = data.iloc[:, 1].tolist()
        nr_vehicles = data.shape[0]
        end_of_week = False
        for day in range(99, 676, 96):
            if count == nr_iterations:
                break
            dayly_data = data.iloc[:, day - 97:day]
            s = env.reset(dayly_data, vehicle_list)
            done = False
            if day == 675:
                end_of_week = True
            counter = 0
            while not done:
                print(counter)
                # always charge all vehicles
                act = np.ones(nr_vehicles)
                s, rew, dayly_data, done, _ = env.step(act, dayly_data, end_of_week)
                env.render(dayly_data)
                print("Reward in this step:", rew)
                counter += 1
            env.episode_summary_statistics(dayly_data)
            count += 1

