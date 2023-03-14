import numpy as np
import gymnasium as gym
from gymnasium import spaces

# DEFINE CONSTANTS
POWER = 11  # kw
DT = 0.25  # per default we do time steps of 1 hour
# TODO: possibly use shorter episodes?
EPISODE_LEN = int(24 / DT)
# TODO: model some electicity price curve that (randomly) changes over the day
electricity_price = np.ones(EPISODE_LEN)


class CarsharingEnv(gym.Env):
    def __init__(self, nr_vehicles=5, dt=DT):
        """
        Parameters
        ----------
        nr_vehicles : int, optional
            Number of deployed vehicles, by default 5
        dt : float, optional
            Time step size, by default 0.25 (a quarter hour)
        """
        self.nr_vehicles = nr_vehicles
        self.dt = dt
        # TODO: load reservation data
        # set battery capacity for each vehicle # TODO: load this from the database as well
        self.battery_capacities = [75 for _ in range(nr_vehicles)]
        # State: two parts:
        # 1) binary value for each car whether it's at the station or not (0 or 1)
        # 2) SOC for each vehicle (between 0 and 1)
        self.observation_space = spaces.Box(np.zeros(nr_vehicles * 2), np.ones(nr_vehicles * 2), dtype=np.float32)
        self.state_dim = self.observation_space.shape[0]
        # Action: discrete, but one for each vehicle --> MultiDiscrete with 3 possible actions for each of them
        # Action space: 0 = do nothing, 1 = charge, 2 = discharge
        self.action_space = spaces.MultiDiscrete([3 for _ in range(nr_vehicles)])
        # TODO (later): simulate the power grid demand for energy (energy expected from Mobility after bidding-phase)

    def reset(self):
        # set time to 0
        self.t = 0
        # resets all vehicles to a random configuration (TODO: set this to a random day in the real bookings)
        self.state = np.random.rand(self.state_dim)
        # binarize first part of the state
        self.state[: self.nr_vehicles] = np.around(self.state[: self.nr_vehicles])
        return self.state

    def step(self, action):
        # TODO: check whether car is booked and update first part of state

        # update SOC according to action
        # energy (in kWh) charged per vehicle:
        energy_to_charge = action * POWER * self.dt
        # can only charge until capacity is reached!
        # TODO: this only works for charging so far (action = 1), but not for discharging (action = 2)
        currently_charged = self.battery_capacities * self.state[self.nr_vehicles :]
        energy_to_charge = np.minimum(energy_to_charge, self.battery_capacities - currently_charged)
        # can only charge cars that are at the station! --> use boolean index to set these to False
        not_at_station = np.around(1 - self.state[: self.nr_vehicles]).astype(bool)
        energy_to_charge[not_at_station] = 0

        # update state
        self.state[self.nr_vehicles :] += energy_to_charge / self.battery_capacities

        # compute reward: NOTE: atm it's just the negative reward for charging! TODO: Add bookings
        rew = np.sum(-1 * energy_to_charge * electricity_price[self.t % EPISODE_LEN])

        done = True if self.t == EPISODE_LEN else False
        # update time step
        self.t += 1

        return self.state, rew, done, {}

    def render(self):
        # TODO: find some way to visualize this, e.g. barplot how many vehicles are at the station, histogram how much
        # the vehicles are charged
        print(f"-------- State at time {self.t} --------")
        print("vehicle ID:", [i for i in range(self.nr_vehicles)])
        print("at station:", self.state[: self.nr_vehicles])
        print("SOC       :", np.around(self.state[self.nr_vehicles :], 2))

