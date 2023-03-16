import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

from plotting import show_soc, my_cmap

### DEFINE CONSTANTS ###
# power for charing car
POWER_charge = 11  # kw
# power during driving
POWER_driving = 10  # kw
# power during energy delivery
POWER_dicharge = 11  # kw
# average trip distance during 1h reservation
average_distance = 50  # km
# revenue per driven km
revenue_km = 0.7  # CHF
# revenue per booked h
revenue_h = 2.0  # CHF
# penalty costs (service should be reliable, otherwise customers get lost and change transport mode)
cancellation_penalty = 100  # CHF
# delta t, time step size
DT = 0.25  # h
# episode length of one day
EPISODE_LEN = int(24 / DT)
# EPISODE_LEN = 12 # --> only used for test dataset

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
        self.vehicles_id = []
        self.nr_vehicles = None
        self.dt = dt
        self.reward_list_trips = []
        self.reward_list_charging = []
        self.reward_list_discharging = []
        self.reward_list_cancellation_penalty = []
        self.reward_list = []
        # TODO: load reservation data

        # set battery capacity for each vehicle # TODO: load this from the database as well

        # --> how select battery capacity?

        self.battery_capacities = [75 for _ in range(nr_vehicles)]
        # State: two parts:
        # 1) location of vehicle: station number (1000-5000), or needed charging level (0-100) if start of trip,
        # reservation number ( x'xxx'xxx) druing trip, or -1 if car not available
        # 2) SOC for each vehicle (between 0 and 1)
        self.observation_space = gym.spaces.Box(
            low=np.array([-1, 0] * nr_vehicles),  # Lower bounds for each dimension
            high=np.array([9999999, 1] * nr_vehicles)  # Upper bounds for each dimension
        )

        self.state_dim = self.observation_space.shape[0]
        # Action: discrete, but one for each vehicle --> MultiDiscrete with 3 possible actions for each of them
        # Action space: 0 = do nothing, 1 = charge, 2 = discharge
        self.action_space = spaces.MultiDiscrete([3 for _ in range(nr_vehicles)])
        # TODO (later): simulate the power grid demand for energy (energy expected from Mobility after bidding-phase)

    def reset(self, dayly_data, vehicle_list):
        # set time to 0
        self.t = 0
        self.reward_list_trips = []
        self.reward_list_charging = []
        self.reward_list_discharging = []
        self.reward_list_cancellation_penalty = []
        self.reward_list = []
        # resets all vehicles to beginning of day
        self.vehicles_id = vehicle_list  #
        self.nr_vehicles = len(self.vehicles_id)  # total number of cars
        car_SOC = np.random.rand(self.nr_vehicles)  # random initial charging levels
        car_locations = dayly_data.iloc[:,
                        0].values  # location of vehicle: station number (1000-5000), or needed charging level (0-100) if start of trip,
        # reservation number ( x'xxx'xxx) druing trip, or -1 if car not available

        # create inital state
        self.state = np.concatenate([car_locations, car_SOC])

        return self.state

    def step(self, action, dayly_data, end_of_week):
        # TODO: check whether car is booked and update first part of state

        # cars cannot be returned from trips with negative loaded battery, set them zero
        self.state[self.nr_vehicles:][self.state[self.nr_vehicles:] < 0] = 0

        # get current car locations
        car_locations = dayly_data.iloc[:, self.t + 1].values

        # update state of car locations
        self.state[:self.nr_vehicles] = car_locations

        # check if car booked or not (start of booking)
        booking_start = (self.state[:self.nr_vehicles] >= 0) & (self.state[:self.nr_vehicles] <= 100)

        # check if engough energy charged for booked trip (otherwise trips needs to be cancelled)
        energy_needed = np.zeros_like(booking_start)
        energy_needed[booking_start == True] = 1
        energy_needed = car_locations * energy_needed
        difference_energy = self.state[self.nr_vehicles:] * energy_needed - energy_needed
        cancelled_trips = difference_energy < 0
        # get indices of cars with cancelled reservation because of low battery level
        indices = np.where(cancelled_trips)[0]
        # count number of cancellation, used for reward penalty
        penalty_counter = len(indices)
        # remove booking from dataset (if reservation is cancelled, the car is again available)
        for index in indices:
            found = False
            last_station = 0
            t = self.t
            car_location = -1
            while not found:
                # search last station of car in bookings, replace reservation number with last location
                if t > 0:
                    car_location = dayly_data.iloc[index, t]
                    if car_location >= 1000 and car_location <= 6000:
                        found = True
                else:
                    # if no station back in time, set car location to -1
                    found = True
                t = t - 1
            # update current state of car with actual location
            self.state[:self.nr_vehicles][index] = car_location
            changed = False
            # remove reservation from dataset
            for col_index, col_data in enumerate(dayly_data.iloc[index]):
                if col_index >= self.t and dayly_data.iloc[index, col_index] >= 1000000:
                    dayly_data.iat[index, col_index] = car_location
                    changed = True
                elif col_index >= self.t and dayly_data.iloc[index, col_index] < 1000000 and changed == True:
                    break

        # check if car on trip
        on_trip = (self.state[:self.nr_vehicles] >= 1000000) | (self.state[:self.nr_vehicles] >= 0) & (
                    self.state[:self.nr_vehicles] <= 100)

        # check if car in service or not (e.g., defect)
        not_in_unse = self.state[:self.nr_vehicles] < 0

        # check if car usable for charging or discharging
        not_chargable = (self.state[:self.nr_vehicles] < 1000) | (self.state[:self.nr_vehicles] > 6000)

        # update SOC according to action
        # filter charging action
        charging_action = np.zeros_like(action)
        discharging_action = np.zeros_like(action)
        charging_action[action == 1] = 1
        discharging_action[action == 2] = 1
        # energy (in kWh) charged/discharged per vehicle:
        energy_to_charge = charging_action * POWER_charge * self.dt
        energy_to_discharge = discharging_action * POWER_dicharge * self.dt
        # can only charge until capacity is reached
        # can only discharge until battery is empty
        currently_charged = self.battery_capacities * self.state[self.nr_vehicles:]
        energy_to_charge = np.minimum(energy_to_charge, self.battery_capacities - currently_charged)
        energy_to_discharge = -np.minimum(energy_to_discharge, currently_charged)
        # can only charge cars that are at the station
        energy_to_charge[not_chargable] = 0
        energy_to_discharge[not_chargable] = 0
        # energy loss by customer trips
        energy_consumption_driving = np.ones((self.nr_vehicles,)) * - POWER_driving * self.dt
        energy_consumption_driving[~on_trip] = 0
        # total SOC difference for each car
        energy_difference = np.add(np.add(energy_to_charge, energy_to_discharge), energy_consumption_driving)
        # update SOC state
        self.state[self.nr_vehicles:] += energy_difference / self.battery_capacities
        # compute reward:
        # charging costs
        rew_charging = np.sum(-1 * energy_to_charge * electricity_price[self.t % EPISODE_LEN])
        # discharging revenue
        max_index = np.argmax(energy_to_discharge)
        rew_discharging = -np.sum(1 * energy_to_discharge * electricity_price[self.t % EPISODE_LEN])
        # trip revenue (duration + driven distance)
        on_trip_numerical = np.where(on_trip, 1, 0)
        trip_reward_distance = np.sum(on_trip_numerical * average_distance * self.dt * revenue_km)
        trip_reward_duration = np.sum(on_trip_numerical * self.dt * revenue_h)
        # penalty for cancelled reservation, because of flat batteries
        reward_cancellations = -cancellation_penalty * penalty_counter
        # total reward
        rew = rew_charging + rew_discharging + trip_reward_distance + trip_reward_duration + reward_cancellations
        self.reward_list.append(rew)
        self.reward_list_trips.append(trip_reward_distance + trip_reward_duration)
        self.reward_list_charging.append(rew_charging)
        self.reward_list_discharging.append(rew_discharging)
        self.reward_list_cancellation_penalty.append(reward_cancellations)

        if end_of_week is True:
            done = True if self.t == (EPISODE_LEN - 2) else False
        else:
            done = True if self.t == (EPISODE_LEN - 1) else False
        # update time step
        self.t += 1

        return self.state, rew, dayly_data, done, {}

    def render(self, dayly_data):
        # TODO: find some way to visualize this, e.g. barplot how many vehicles are at the station, histogram how much
        # the vehicles are charged
        print(f"-------- State at time {self.t} --------")
        # print("vehicle ID:", self.vehicles_id)
        # print("at station:", self.state[: self.nr_vehicles])
        # print("SOC       :", np.around(self.state[self.nr_vehicles :], 2))
        timestamp = dayly_data.columns[self.t]

        # plot location of vehicles
        on_trip = (self.state[:self.nr_vehicles] >= 1000000) | (self.state[:self.nr_vehicles] >= 0) & (
                    self.state[:self.nr_vehicles] <= 100)
        at_station = (self.state[:self.nr_vehicles] >= 1000) & (self.state[:self.nr_vehicles] <= 6000)
        count_trip = np.sum(on_trip == True)
        count_at_station = np.sum(at_station == True)
        categories = ["At Station", "On Trip"]
        values = [count_at_station, count_trip]
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.bar(categories, values)
        plt.ylabel("Numer of Cars")
        plt.title("Location of Cars ({})".format(timestamp))
        plt.xticks(rotation=90)

        plt.show()

    def render_soc(self):
        soc = self.state[self.nr_vehicles:]
    
        side_size = int(np.sqrt(self.nr_vehicles))
        img_w, img_h = side_size, side_size+1
        assert img_w * img_h > self.nr_vehicles
        img = np.zeros((img_w * img_h, 4))
        at_station = np.zeros(img_w * img_h)
        at_station[:self.nr_vehicles] = self.state[:self.nr_vehicles]

        img[:self.nr_vehicles] = np.array([my_cmap(val) for val in soc])
        img[at_station < 0] = np.array([0.8, 0.8, 0.8, 1])
        img[at_station > 10000000] = np.array([0.8, 0.2, 0.1, 1])
        
        show_soc(img.reshape((img_h, img_w, 4)))

    def episode_summary_statistics(self, dayly_data):
        # get data
        date = dayly_data.columns[1][:10]
        rew_trips = sum(self.reward_list_trips)
        rew_charging = sum(self.reward_list_charging)
        rew_discharging = sum(self.reward_list_discharging)
        rew_cancellation = sum(self.reward_list_cancellation_penalty)
        rew_total = sum(self.reward_list)

        # plot reward over time as lineplot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot([s[-8:-3] for s in dayly_data.columns[1:]], self.reward_list)
        print([s[-8:-3] for s in dayly_data.columns[1:]])
        # Format the x-axis labels
        plt.xticks(rotation=90)
        xtick_labels = ax.get_xticklabels()
        for i, label in enumerate(xtick_labels):
            if i % 2 != 0:
                label.set_visible(False)
        plt.xlabel("Time")
        plt.ylabel("Reward (CHF)")
        plt.title("Reward over Time")
        plt.show()

        # composition of total reward of episode
        categories = ["Trip Revenue", "Charging Costs", "Discharging Revenue", "Cancellation Penalty", "Total Reward"]
        values = [rew_trips, rew_charging, rew_discharging, rew_cancellation, rew_total]
        fig, ax = plt.subplots()
        ax.bar(categories, values)
        plt.ylabel("Reward [CHF]")
        plt.title("Episodes Reward ({})".format(date))
        plt.xticks(rotation=90)
        plt.show()