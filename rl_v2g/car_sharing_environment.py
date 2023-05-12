import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import bisect
import random
import math
from rl_v2g.plotting import show_soc, my_cmap


class CarsharingEnv(gym.Env):
    def __init__(self, stations, vehicle_information, daily_data, reservations, electricity_price, timesteps_since_start, v2g_price = None, planned_reservations=None,
              planned_durations=None, soc_initial_low=0.5, soc_initial_high=1, max_charging_power=11,
                 episode_len=24, dt=0.25, last_timestep = 55392,
                 random_seed_number=122, cancellation_penalty=100, penalty_per_kwh=0.0, v2g=True,
                 v2g_demand_event_min=500, v2g_demand_event_max=500, v2g_max_duration=2.0, v2g_penalty=499,
                 v2g_probability_charging_event=0.5, v2g_probability_discharging_event=0.5,
                 v2g_morning_time_period=[6.0, 9.0, 11.00], v2g_noon_time_period=[11.25, 14.0, 15.75],
                 v2g_evening_time_period=[16.0, 20.0, 22.0],
                 planned_bookings=True, precomputed_bookings=True, max_distance_car_assingment=1000,
                 plot_state_histogram=False, plot_state_animation=False, RL = False):
        """
        Initialization of simulation environment for car-sharing charging and/or vehicle-to-grid (V2G) optimization.

        Parameters
        ----------
        stations: Geopandas Geodataframe
            Locations of car-sharing stations, including a distinct "station_no" attribute with station ID.
        vehicle_information: Pandas Dataframe
            Includes the features "vehicle_no", "charge_power", "battery_capacity", and "vehicle_category" for each car.
        daily_data: Pandas DataFrame
            Contains the car trips over the day.
        reservations: Pandas DataFrame
            Includes the features "syscreatedate_time_discrete", "vehicle_no", "reservationfrom_time_discrete", and "reservation_duration" for each reservation.
        electricity_price: Numpy ndarray
            Contains electricity prices for each timestamp.
        timesteps_since_start: int
            Timestamp since start of simulation before current day.
        v2g_price: Numpy ndarray, optional
            Numpy array of legngth self.episode_len containing prices in CHF per kwh, by default None.
        planned_reservations: Pandas DataFrame, optional
            Timestamps of next planned reservations for each car, by default None.
        planned_durations: Pandas DataFrame, optional
            Durations of next planned reservations for each car, by default None.
        soc_initial_low: float, optional
            Lower bound (0-1) for initial SOC used for the reset of the environment, by default 0.5.
        soc_initial_high: float, optional
            Upper bound (0-1) for initial SOC used for the reset of the environment, by default 0.5.
        max_charging_power: int, optional
            Maximum charging power.
        episode_len: int, optional
            Length of one episode in hours, by default 24.
        dt: float, optional
            Time step size in hours, by default 0.25 (a quarter hour).
        cancellation_penalty: int, optional
            Maximum penalty in CHF for a booking cancelled due to not enough charged battery, by default 100.
        last_timestep: int, optional
            Last timestep of simulation, by defualt 55392.
        random_seed_number: int, optional
            Used for fixing random components in environment for reproducable results, by default 42.
        penalty_per_kwh: int, optional
            Penalty in CHF/kWh for total negative energy difference between beginning and ending of episode, by default 0.25.
        v2g: boolean, optional
            Boolean indicating if V2G events take place, by default True.
        v2g_demand_event_min: int, optional
            Minimum energy demand during V2G event in kWh per time step (dt), by default 500.
        v2g_demand_event_max: int, optional
            Maxiumunm energy demand during V2G event in kwh per time step (dt), by default 500.
        v2g_max_duration: int, optional
            Maximum duration of V2G charging or discharging event in hours, by default 3.
        V2G_penalty: int, optional
            Penalty in CHF if agent charges/discharges less energy than specified in "v2g_demand_event" during V2G event, by default 10000.
        v2g_probability_charging_event: float, optional
            Probability that a charging event will take place around noon, by default 0.5.
        v2g_probability_discharging_event: float, optional
            Probability that a discharging event will take place in the morning or evening, by default 0.5.
        v2g_morning_time_period: list, optional
            Time period must end before noon.
            List containing: 1) first possible time (hours) for starting v2g discharging event in the morning, by default 6.0 (6 AM).
                             2) last possible time (hours) for starting v2g discharging event in the morning, by default 9.0 (9 AM).
                             3) last possible timestamp for v2g discharging operation, by default 10.75 (10:45 AM).
        v2g_noon_time_period: list, optional
            List containing: 1) first possible time (hours) for starting v2g charging event at noon, by default 11.0 (11 AM).
                             2) last possible time (hours) for starting v2g charging event at noon, by default 14.0 (2 PM).
                             3) last possible timestamp for v2g charging operation, by default 16.0 (4 PM).
        v2g_evening_time_period: list, optional
            Time period must start after noon.
            List containing: 1) first possible time (hours) for starting v2g discharging event in the evening, by default 16.25 (4:15 PM).
                             2) last possible time (hours) for starting v2g discharging event in the evening, by default 19.0 (7 PM).
                             3) last possible timestamp for v2g charging operation, by default 24 (00:00 AM, next day).
        planned_bookings: boolean, optional
            Boolean indicating whether there are planned bookings in the environment; otherwise, all bookings are spontaneous, by default True.
        precomputed_bookings: boolean, optional
            Boolean indicating whether the planned bookings in the environment are precomputed (for massive speed up), by default True.
        max_distance_car_assingment: int, optional
            Maximum search distance in meter for car-assingment problem, by default 5000.
        plot_state_histogram: boolean, optional
            Plot current state of enviorment as histogram or not, by default True.
        plot_state_animation: boolean, optional
            Plot current state of enviorment as animation for each car or not, by default True.
        RL: boolean
            Whether to environment for reinforcement learning or not.
        ----------
        Observation space: 5 parts:
        1) Location for each car (four options):
            - Station number (1000-5000)
            - Reservation number (2x'xxx'xxx) during trip
            - Reservation number (3x'xxx'xxx) during relocation
            - -1 if car is not available
        2) State of charge (SOC) for each vehicle (between 0 and 1).
        3) Timestamp of the next planned booking for each car (discrete between 0 and self.episode_len).
        4) Duration of the next planned booking (measured in the number of time steps of length self.dt).
        5) Binary variable indicating the occurrence of a vehicle-to-grid (V2G) event.
        6) Current time (discrete value indicating the time in self.dt since first day of simulation)
        Remark: States 3) and 4) are only included if the variable "planned_bookings" is True.
        ----------
        Action space: 3 actions for each car:
        1) 0 = do nothing.
        2) 1 = charging.
        3) 3 = discharging (V2G).
        """

        # environment settings
        self.dt = dt
        self.episode_len = int(episode_len / self.dt)
        self.planned_bookings = planned_bookings
        self.precomputed_bookings = precomputed_bookings
        self.max_distance_car_assingment = max_distance_car_assingment
        self.plot_state_histogram = plot_state_histogram
        self.plot_state_animation = plot_state_animation
        self.random_seed_number = random_seed_number
        self.last_timestep = last_timestep

        # stations in system
        self.stations = stations

        # vehicle information
        self.vehicles_id = vehicle_information["vehicle_no"]
        self.nr_vehicles = len(self.vehicles_id)
        self.chariging_power = np.full_like(vehicle_information["charge_power"],
                                            np.minimum(vehicle_information["charge_power"], max_charging_power))
        self.battery_capacities = vehicle_information["battery_capacity"]
        self.vehicle_type = vehicle_information["vehicle_category"]
        self.soc_initial_low = soc_initial_low
        self.soc_initial_high = soc_initial_high

        # V2G
        self.v2g = v2g
        self.v2g_demand_event_min = v2g_demand_event_min
        self.v2g_demand_event_max = v2g_demand_event_max
        self.v2g_max_duration = v2g_max_duration / self.dt
        self.v2g_penalty = v2g_penalty
        self.v2g_probability_charging_event = v2g_probability_charging_event
        self.v2g_probability_discharging_event = v2g_probability_discharging_event
        self.v2g_morning_time_period = v2g_morning_time_period
        self.v2g_noon_time_period = v2g_noon_time_period
        self.v2g_evening_time_period = v2g_evening_time_period

        # penalties
        self.cancellation_penalty = cancellation_penalty
        self.penalty_per_kwh = penalty_per_kwh

        # define observation space:
        if self.planned_bookings is True:
            
            self.locations_of_vehicles_space = spaces.Box(low=-1, high=31000000, shape=(self.nr_vehicles,), dtype=np.float64)
            self.soc_of_vehicles_space = spaces.Box(low=0, high=1, shape=(self.nr_vehicles,),dtype=np.float64)
            self.planned_reservations_space = spaces.Box(low=-1, high=self.last_timestep, shape=(self.nr_vehicles,), dtype=np.float64)
            self.planned_reservation_durations_space = spaces.Box(low=-1, high=self.last_timestep, shape=(self.nr_vehicles,), dtype=np.float64)
            self.v2g_event_space = spaces.Box(low=-self.v2g_demand_event_max, high=self.v2g_demand_event_max, shape=(1,), dtype=np.float64)
            self.current_time_space = spaces.Box(low=0, high=self.last_timestep, shape=(1,), dtype=np.float64)
            self.observation_space = spaces.Dict({
                'locations_of_vehicles': self.locations_of_vehicles_space,
                'soc_of_vehicles': self.soc_of_vehicles_space,
                'planned_reservations': self.planned_reservations_space,
                'planned_reservation_durations': self.planned_reservation_durations_space,
                'v2g_event': self.v2g_event_space,
                'current_time': self.current_time_space
            })

            self.observation_space = spaces.Box(
            
                low=np.concatenate([
                    self.locations_of_vehicles_space.low, self.soc_of_vehicles_space.low,
                    self.planned_reservations_space.low, self.planned_reservation_durations_space.low,
                    self.v2g_event_space.low, self.current_time_space.low
                ]),
                high=np.concatenate([
                    self.locations_of_vehicles_space.high, self.soc_of_vehicles_space.high,
                    self.planned_reservations_space.high, self.planned_reservation_durations_space.high,
                    self.v2g_event_space.high, self.current_time_space.high
                ]),
                dtype=np.float64,
                shape=(self.nr_vehicles * 4 + 2,)
            )

        else:
            self.locations_of_vehicles_space = spaces.Box(low=-1, high=5000, shape=(self.nr_vehicles,), dtype=np.float64)
            self.soc_of_vehicles_space = spaces.Box(low=0, high=1, shape=(self.nr_vehicles,), dtype=np.float64)
            self.v2g_event_space = spaces.Box(low=-self.v2g_demand_event_max, high=self.v2g_demand_event_max,
                                              shape=(1,), dtype=np.float64)
            self.current_time_space = spaces.Box(low=0, high=self.last_timestep, shape=(1,), dtype=np.float64)

            self.observation_space = spaces.Box(
                low=np.concatenate([
                    self.locations_of_vehicles_space.low, self.soc_of_vehicles_space.low,
                    self.v2g_event_space.low, self.current_time_space.low
                ]),
                high=np.concatenate([
                    self.locations_of_vehicles_space.high, self.soc_of_vehicles_space.high,
                    self.v2g_event_space.high, self.current_time_space.high
                ]),
                dtype=np.float32,
                shape=(self.nr_vehicles * 2 + 2,)
            )

        # define action space
        self.action_space = spaces.MultiDiscrete([3 for _ in range(self.nr_vehicles)])
        # define state boundaries for slicing state array
        # locations upper bound
        self.locations_upper = self.nr_vehicles

        # soc upper bound
        self.soc_upper = 2 * self.nr_vehicles

        if planned_bookings is True:

            # planned reservations, upper bound:
            self.reservation_time_upper = self.nr_vehicles * 3

            # v2g, lower bound:
            self.v2g_lower = self.nr_vehicles * 4

            # v2g, upper bound:
            self.v2g_upper = self.nr_vehicles * 4 + 1

            # time, lower bound:
            self.time_lower = self.nr_vehicles * 4

        else:

            # v2g, lower bound:
            self.v2g_lower = self.nr_vehicles * 2

            # v2g, upper bound:
            self.v2g_upper = self.nr_vehicles * 2 + 1

            # time, lower bound:
            self.time_lower = self.nr_vehicles * 2

        # laod data
        self.load_new_data(daily_data, reservations, electricity_price, timesteps_since_start, v2g_price, planned_reservations,
              planned_durations)

        # reset environment
        self.reset()

    def load_new_data(self, daily_data, reservations, electricity_price, timesteps_since_start, v2g_price = None, planned_reservations=None,
              planned_durations=None):
        """
        Loads data to environment of a new day

        Parameters
        ----------
        daily_data: Pandas DataFrame
            Contains the car trips over the day.
        reservations: Pandas DataFrame
            Includes the features "syscreatedate_time_discrete", "vehicle_no", "reservationfrom_time_discrete", and "reservation_duration" for each reservation.
        electricity_price: Numpy ndarray
            Contains electricity prices for each timestamp.
        timesteps_since_start: int
            Timestamp since start of simulation before current day.
        v2g_price: Numpy ndarray, optional
            Numpy array of legngth self.episode_len containing prices in CHF per kwh, by default None.
        planned_reservations: Pandas DataFrame, optional
            Timestamps of next planned reservations for each car, by default None.
        planned_durations: Pandas DataFrame, optional
            Durations of next planned reservations for each car, by default None.
        """

        self.reset_daily_data = daily_data
        self.reset_reservations = reservations
        self.reset_electricity_price = electricity_price
        self.reset_timesteps_since_start= timesteps_since_start
        self.reset_v2g_price = v2g_price
        self.reset_planned_reservations = planned_reservations
        self.reset_planned_durations = planned_durations


    def reset(self):
        """
        Reset environment to beginning of new episode.

        Returns
        ----------
        self.state: numpy ndarray
            The reset state of the environment at the first time step. It includes the following information for each car:
            1) Location (four options):
                - Station number (1000-5000)
                - Reservation number (2x'xxx'xxx) during trip
                - Reservation number (3x'xxx'xxx) during relocation
                - -1 if the car is not available
            2) State of charge (SOC) for each vehicle (between 0 and 1).
            3) Timestamp of the next planned booking for each car (discrete value between 0 and self.episode_len).
            4) Duration of the next planned booking (measured in the number of time steps of length self.dt).
            5) Binary variable indicating the occurrence of a vehicle-to-grid (V2G) event.
            6) Current time (discrete value indicating the time in self.dt since first day of simulation)
            Note: The information in 3) and 4) is included only if the variable "planned_bookings" is True.
        """

        # assign simulation data
        self.daily_data = self.reset_daily_data

        # assign reservation data
        self.reservations = self.reset_reservations

        # assign electricity prices
        self.electricity_price = self.reset_electricity_price

        # assign planned reservations and the durations
        if self.planned_bookings is True:
            if self.precomputed_bookings is True:
                self.planned_reservations = self.reset_planned_reservations
                self.planned_durations = self.reset_planned_durations

        # set timesteps since start of t = 0
        self.timesteps_since_start = self.reset_timesteps_since_start

        # set time
        self.t = self.reset_timesteps_since_start

        # reset reward lists
        self.reward_list_trips = []
        self.reward_list_charging = []
        self.reward_list_cancellation_penalty = []
        self.reward_list_v2g = []
        self.reward_list = []

        # reset list with processed timestamps
        self.date_list = []

        # reset old state
        self.state_old = np.zeros(self.nr_vehicles * 4 + 1)

        # reset planned reservations:
        self.planned_reservations_car = {key: [] for key in self.vehicles_id}

        if self.v2g is True:
            # get random v2g events
            self.get_random_v2g_events()

            # assing v2g discharging event prices
            self.v2g_price_discharging = self.reset_v2g_price[0]

            # assing v2g charging event prices
            self.v2g_price_charging = self.reset_v2g_price[1]

        # reset changes discrete table
        self.changed_vehicles = {}

        # reset changed reservations
        self.changed_reservations = {key: 0 for key in range(self.nr_vehicles)}

        ### initialize state for t = 0 ###
        # 1) car locations (three options)
        car_locations = self.daily_data.iloc[:, 0 + self.timesteps_since_start % self.episode_len].values

        # 2) SOC (state of charge) ##
        rng = np.random.RandomState(self.random_seed_number + self.t)
        car_SOC = rng.uniform(low=self.soc_initial_low, high=self.soc_initial_high, size=self.nr_vehicles)

        # state 5) Binary V2G event
        v2g_event = np.array([0])

        # state 6) current time
        current_time = np.array([self.t])

        # create final state, (add planned bookings if needed)
        if self.planned_bookings is True:

            # 3) timestamp of next planned reservation
            # 4) Duration of next planned reservation
            if self.precomputed_bookings is True:

                # load precomputed state
                next_reservation = self.planned_reservations.iloc[:,
                                   0 + self.timesteps_since_start % self.episode_len].values
                duration_next_reservation = self.planned_durations.iloc[:,
                                            0 + self.timesteps_since_start % self.episode_len].values

            else:

                # calculate state
                next_reservation, duration_next_reservation = self.update_reservation_state(True)

            # concatinate states 1-5
            self.state = np.concatenate(
                [car_locations, car_SOC, next_reservation, duration_next_reservation, v2g_event, current_time])

            #state = {
            #    'locations_of_vehicles': self.state[:self.locations_upper],
            #    'soc_of_vehicles': self.state[self.locations_upper:self.soc_upper],
            #    'planned_reservations': self.state[self.soc_upper:self.reservation_time_upper],
            #    'planned_reservation_durations': self.state[self.reservation_time_upper:self.v2g_lower],
            #    'v2g_event': self.state[self.v2g_lower:self.v2g_upper],
            #    'current_time': self.state[self.v2g_upper:]
            #}

        else:

            # concatinate states 1-5 without planned boookings (state 3,4)
            self.state = np.concatenate([car_locations, car_SOC, v2g_event, current_time])
            # transform to dict:
            #state = {
            #   'locations_of_vehicles': self.state[:self.locations_upper],
            #   'soc_of_vehicles': self.state[self.locations_upper:self.soc_upper],
            #    'v2g_event': self.state[self.v2g_lower:self.v2g_upper],
            #    'current_time': self.state[self.v2g_upper:]
            #}

        # save energy at beginning of episode (for reward calculation)
        self.energy_beginning = sum(car_SOC * self.battery_capacities)

        print("Reset environment to timestamp: ", self.timesteps_since_start)



        return self.state


    def next_day(self, daily_data, reservations, electricity_price, timesteps_since_start, v2g_price = None, planned_reservations=None,
                 planned_durations=None):
        """
        Loads data for simulation of new day. Does not reset the enviroment

        Parameters
        ----------
        daily_data: Pandas DataFrame
            Contains the car trips over the day.
        reservations: Pandas DataFrame
            Includes the features "syscreatedate_time_discrete", "vehicle_no", "reservationfrom_time_discrete", and "reservation_duration" for each reservation.
        electricity_price: Numpy ndarray
            Contains electricity prices for each timestamp.
        timesteps_since_start: int
            Timestamp since start of simulation before current day.
        v2g_price: Numpy ndarray, optional
            Numpy array of legngth self.episode_len containing prices in CHF per kwh, by default None.
        planned_reservations: Pandas DataFrame, optional
            Timestamps of next planned reservations for each car, by default None.
        planned_durations: Pandas DataFrame, optional
            Durations of next planned reservations for each car, by default None.
        """

        # assign simulation data
        self.daily_data = daily_data

        # assign reservation data
        self.reservations = reservations

        # assign electricity prices
        self.electricity_price = electricity_price

        if self.v2g is True:
            # assing v2g discharging event prices
            self.v2g_price_discharging = v2g_price[0]

            # assing v2g charging event prices
            self.v2g_price_charging = v2g_price[1]

            # get random V2G events
            self.get_random_v2g_events()

        # assign planned reservations and the durations
        if self.planned_bookings is True:
            if self.precomputed_bookings is True:
                self.planned_reservations = planned_reservations
                self.planned_durations = planned_durations

        # set timesteps since start of t = 0
        self.timesteps_since_start = timesteps_since_start


    def get_random_v2g_events(self):
        """
        Samples random V2G events during the day. A discharging event can take place in the morning and evening, a charging
        event at noon.
        """

        # define variables
        self.v2g_discharing_morning_start = -1
        self.v2g_discharing_morning_end = -1
        self.v2g_charing_noon_start = -1
        self.v2g_charing_noon_end = -1
        self.v2g_discharing_evening_start = -1
        self.v2g_discharing_evening_end = -1
        self.v2g_demand_event = [0,0,0]

        # use fixed random seed
        random.seed(self.random_seed_number + self.t)

        # sample random v2g event
        # morning discharging event:
        if int(random.random() < self.v2g_probability_discharging_event) == 1:
            # get random start time
            self.v2g_discharing_morning_start = round(random.uniform(self.v2g_morning_time_period[0],
                                                                     self.v2g_morning_time_period[
                                                                         1]) / 0.25) * 0.25 / self.dt

            # get random end time
            self.v2g_discharing_morning_end = min(self.v2g_morning_time_period[2] / self.dt,
                                                  self.v2g_discharing_morning_start + round(random.uniform(0.25,
                                                                                                           self.v2g_max_duration) / 0.25) * 0.25 / self.dt)

            # get energy to deliver in kwh
            self.v2g_demand_event[0] = random.uniform(self.v2g_demand_event_min, self.v2g_demand_event_max)

        # noon charging event
        if int(random.random() < self.v2g_probability_charging_event) == 1:
            # get random start time
            self.v2g_charing_noon_start = round(
                random.uniform(self.v2g_noon_time_period[0], self.v2g_noon_time_period[1]) / 0.25) * 0.25 / self.dt

            # get random end time
            self.v2g_charing_noon_end = min(self.v2g_noon_time_period[2] / self.dt,
                                            self.v2g_charing_noon_start + round(random.uniform(0.25,
                                                                                               self.v2g_max_duration) / 0.25) * 0.25 / self.dt)
            # get energy to take in kwh
            self.v2g_demand_event[1] = random.uniform(-self.v2g_demand_event_max, -self.v2g_demand_event_min)

        # evening discharging event
        if int(random.random() < self.v2g_probability_discharging_event) == 1:
            # get random start time
            self.v2g_discharing_evening_start = round(random.uniform(self.v2g_evening_time_period[0],
                                                                     self.v2g_evening_time_period[
                                                                         1]) / 0.25) * 0.25 / self.dt

            # get random end time
            self.v2g_discharing_evening_end = min(self.v2g_evening_time_period[2] / self.dt,
                                                  self.v2g_discharing_evening_start + round(random.uniform(0.25, self.v2g_max_duration) / 0.25) * 0.25 / self.dt)

            # get energy to deliver in kwh
            self.v2g_demand_event[2] = random.uniform(self.v2g_demand_event_min, self.v2g_demand_event_max)
            
        
            
        

        return


    def update_reservation_state(self, reset):
        """
         Parameters
        ----------
        reset: boolean, optional
            Boolean indicating whether to reset the environment or not.

        Returns
        ----------
        next_reservation : numpy ndarray
            Timestamp of the next planned booking for each car (discrete value between 0 and and last timestep of data).
        duration_next_reservation : numpy ndarray
            Duration of the next planned booking (measured in the number of time steps of length self.dt).
        """

        # initalize variables during reset of environment
        if reset is True:
            time_ = 0
            next_reservation = np.ones(self.nr_vehicles) * -1
            duration_next_reservation = np.ones(self.nr_vehicles) * -1

        # get current state of next planned reservations (timestamp and duration)
        else:
            time_ = self.t + 1
            next_reservation = self.state[self.soc_upper: self.reservation_time_upper]
            duration_next_reservation = self.state[self.reservation_time_upper:self.v2g_lower]

        # filter reservations:
        # bookings before t = 0 have value syscreatedate_time_discrete = 0
        reservations_t = self.reservations[self.reservations["syscreatedate_time_discrete"] == time_]
        reservations_t.set_index(['vehicle_no'], inplace=True, drop=False)

        # iterate over all vehicles, search planned reservations
        count = 0
        for vehicle_id in self.vehicles_id:

            # get reservation of car
            current_car = reservations_t[reservations_t["vehicle_no"] == vehicle_id]

            # remove reservations in past
            finish = False
            while finish is False:
                if self.planned_reservations_car[vehicle_id] and int(
                        self.planned_reservations_car[vehicle_id][0][0]) <= self.t:
                    self.planned_reservations_car[vehicle_id] = self.planned_reservations_car[vehicle_id][1:]
                else:
                    finish = True

            # if no new reservation found
            if current_car.empty:

                # if current state's reservation is in the past
                if next_reservation[count] <= self.t and next_reservation[count] != -1:

                    # check for planned reservations saved in the past
                    if self.planned_reservations_car[vehicle_id]:

                        # save reservation timestamp
                        next_reservation[count] = self.planned_reservations_car[vehicle_id][0][0]

                        # save reservation duration
                        duration_next_reservation[count] = self.planned_reservations_car[vehicle_id][0][1]

                    else:

                        # assign -1 for no planned reservation
                        next_reservation[count] = -1
                        duration_next_reservation[count] = -1

                count += 1
                continue

            # iterate over newly found reservations
            for i in range(0, len(current_car)):
                # get timestamp and duration of next planned reservation
                reservation_time = current_car["reservationfrom_time_discrete"].iloc[i]
                reservation_duration = current_car["reservation_duration"].iloc[i]

                # save found reservation environment in dict
                new_reservation = [reservation_time, reservation_duration]

                # remain order in dict (nearest reservation in first index)
                index = bisect.bisect_left([sublist[0] for sublist in self.planned_reservations_car[vehicle_id]],
                                           new_reservation[0])

                # save timestamp and duration of new reservations in dict
                self.planned_reservations_car[vehicle_id].insert(index, new_reservation)

            # assign next reservation
            if self.planned_reservations_car[vehicle_id][0]:
                # get timestamp of next reservation
                next_reservation[count] = self.planned_reservations_car[vehicle_id][0][0]

                # get reservation duration of next reservation
                duration_next_reservation[count] = self.planned_reservations_car[vehicle_id][0][1]

            count += 1

        return next_reservation, duration_next_reservation

    def cancel_trips_low_SOC(self):
        """
        Checks if car have enough SOC for serving trip. If not, the reservation will assinged to a alternative available car
        if possible. In those cases, a distant dependent penalty factor will be calculated.
        ----------
        Returns
        ----------
        penalty_counter: int
            Total factor of penalty for timestep.
        """
        # check if car booked or not (start of booking)
        on_trip = self.state[:self.locations_upper] > 1000000
        on_trip_cond2 = self.state[:self.locations_upper] <= 30000000
        trip_start = self.state[:self.locations_upper] != self.state_old[:self.locations_upper]
        result = on_trip & trip_start & on_trip_cond2

        # array will be filled with 0 for all car that do not start a new trip, otherwise required SOC value
        soc_needed = np.zeros(len(result))

        # get reservation numbers for cars starting trips
        reservation_nos = self.state[:self.locations_upper][result]
        trip_indices = result.nonzero()[0]

        # get required SOC values for relevant cars
        reservations_step = self.reservations.copy()
        soc_needed[trip_indices] = reservations_step.set_index("reservation_no").loc[reservation_nos][
                                       "required_soc"].values / 100

        # calculate difference between SOC of car at the moment and required SOC for the trip
        difference_energy = self.state[self.locations_upper:self.soc_upper] - soc_needed

        # boolean array, True for cars with to low SOC for trip (those reservations will be cancelled)
        cancelled_trips = difference_energy < 0

        # booolean array, True for cars without cancelled trip
        not_cancelled_trips = np.logical_not(cancelled_trips)

        # convert to integer array
        not_cancelled_trips_int = not_cancelled_trips.astype(int)

        # calculate new SOC of cars and add result to state (previous timestamps SOC minus SOC needed for trip)
        self.state[self.locations_upper:self.soc_upper] = (
                    self.state[self.locations_upper:self.soc_upper] - soc_needed * not_cancelled_trips_int)

        # get indices of cars with cancelled reservation because of low battery level
        indices = np.where(cancelled_trips)[0]

        ### remove cancelled booking from discrete tables (if reservation is cancelled, the car is again available) ###
        ### assign bookings to other cars if possible ###

        # sum penalty factor of cancellations, used for reward penalty
        penalty_counter = 0

        # iterate over indices of cars with cancelled reservations
        for index in indices:
            found = False
            last_station = 0

            # get reservation information of cancelled booking
            reservations_of_car = self.reservations[self.reservations["reservation_no"] == self.state[index]]

            # if the reservation cannot be found
            if len(reservations_of_car) == 0:
                print("")
                print("Warning: cancel_trips_low_SOC()")
                print("Reservation in Discrete table not found in Reservation table")
                print("Car ID")
                print(self.vehicles_id[index])
                print("Reservation_number")
                print(self.state[index])
                print("SOC")
                print(self.state[self.nr_vehicles+index])
                print("SOC needed")
                print(soc_needed[index])
                continue

            # get start location of trip
            car_location = reservations_of_car["start_station_no"].iloc[0]

            # get reservation number
            reservation_no = reservations_of_car["reservation_no"].iloc[0]

            # get duration of reservation
            reservation_duration = reservations_of_car["reservation_duration"].iloc[0]

            # get final drive duration (needed for simulate return of car)
            drive_duration = reservations_of_car["drive_duration"].iloc[0]

            # try assining reservation to other car, get cancellation penalty
            penalty = self.car_assignment(car_location, index, soc_needed[index], reservation_no, reservation_duration,
                                          drive_duration)

            # sum up penalties of all cancelled or reassinged reservations
            penalty_counter += penalty

            # update current state of car with actual location
            self.state[index] = car_location

            # remove reservation of car from simulation
            if index in self.changed_vehicles:
                self.changed_vehicles[index].insert(0, (car_location, self.t + 1 + drive_duration, reservation_no))
            else:
                self.changed_vehicles[index] = [(car_location, self.t + 1 + drive_duration, reservation_no)]

        # return total cancellation penalty
        return penalty_counter

    def car_assignment(self, station_id, index_car, soc_needed, reservation_no, reservation_duration, drive_duration,
                       update_soc=True):
        """
        Search alternative, available cars in neigbourhood of cancelled reservation's car.

        Parameters
        ----------
        station_id: int
            ID of station with the cancelled reservation.
        index_car: int
            Index of the car with the cancelled reservaton
        soc_needed: int
            Minimum needed SOC for cancelled trip.
        reservation_no: int
            Reservation number of cancelled reservation.
        reservation_duration: int
            Duration of cancelled reservation, measured in timesteps of self.dt.
        drive_duration: int
            Final drive duration of reservation, measured in timesteps of self.dt.
        update_soc: boolean
            Boolean indicating whether to update SOC state of car or not.

        Returns
        ----------
        penalty_counter: int
            Penalty of single cancelled reservation
        """

        # create local copy of station table
        # stations_processing = self.stations.copy()

        # create spatial index
        # print(stations_processing.has_sindex)

        # get geodata of station with cancelled reservation
        station = self.stations[self.stations["station_no"] == station_id].copy()

        # create spatial index
        station.sindex

        # if station location not available
        if station.empty:
            # return maximum penalty
            penalty = 1

            return penalty

        # get car type of cancelled car
        car_type_cancelled = self.vehicle_type[index_car]

        # query spatial index to get all geometries within distance of self.max_distance_car_assingment
        neighbours = list(self.stations.sindex.query(station["geom"].iloc[0].buffer(self.max_distance_car_assingment),
                                                     predicate='intersects'))

        # create a new dataframe with only the stations within the specified distance
        stations_within_distance = self.stations.iloc[neighbours].copy()

        # calculate distance to each neighbour and sort by distance
        stations_within_distance['distance'] = stations_within_distance.geometry.distance(station["geom"].iloc[0])

        # sort stations by distance
        stations_sorted = stations_within_distance.sort_values('distance')

        # iterate over stations for finding alternative cars
        solution_found = False
        for index_stations, row in stations_sorted.iterrows():

            # get alternative station ID (includes same station as cancelled booking in first iteration)
            alternative_station_ID = row['station_no']

            # get IDs of cars at nearest found station
            cars_at_found_station = self.state[:self.locations_upper] == alternative_station_ID
            indices = np.where(cars_at_found_station)[0]

            # iterate over found cars at alternative station
            for index in indices:
                conflicting_reservations = False

                # check for conflicting reservations, if planned bookings available
                if self.planned_bookings is True:

                    # get timestamp of next planned reservations
                    next_reservation_car = self.state[self.soc_upper + index]

                    # check if conflicting reservation found
                    if next_reservation_car != -1 and next_reservation_car >= self.t + 1 and next_reservation_car <= self.t + 1 + reservation_duration:
                        conflicting_reservations = True

                # if alternative car available (no planned, conflicting reservation yet) and SOC high enough for trip
                if conflicting_reservations is False and (
                        self.state[self.locations_upper:self.soc_upper][index] - soc_needed) >= 0 and \
                        self.state[:self.locations_upper][index] >= 1000 and self.state[:self.locations_upper][
                    index] <= 1000000 and car_type_cancelled == self.vehicle_type[index]:
                    solution_found = True
                    changed = False

                    # calculate distance weighted penalty (zero if car aviailable at the same station)
                    penalty = row['distance'] / self.max_distance_car_assingment

                    # update future simulation of found car
                    if index in self.changed_vehicles:
                        self.changed_vehicles[index].insert(0, (
                        reservation_no, self.t + 1 + drive_duration, alternative_station_ID))
                    else:
                        self.changed_vehicles[index] = [
                            (reservation_no, self.t + 1 + drive_duration, alternative_station_ID)]

                    if update_soc is True:
                        # remove used SOC during trip from vehicle state
                        self.state[self.locations_upper + index] = (
                                    self.state[self.locations_upper + index] - soc_needed)

                        # update state
                    self.state[index] = reservation_no

                    # stopp searching for alternative car if a solution is already found
                    break

            if solution_found is True:
                break

        # return maximum penalty if no alternative found
        if solution_found is False:
            penalty = 1

            # return penalty of cancellation
        return penalty

    def charging(self, action, not_chargable):
        """
        Calculates energy to charge for each car, depending on action and current SOC.

        Parameters
        ----------
        action: numpy ndarray
            Values: 0 for do nothing, 1 for charging, and 2 for discharging
        not_chargable: numpy ndarray
            Boolean for each car, if chargable (at the station) or not (on trip or not implemented)
        Returns
        ----------
        energy_to_charge: numpy ndarray
            Energy to charge for each car.
        """

        # filter charging action
        charging_action = np.zeros_like(action)
        charging_action[action == 1] = 1

        # energy (in kWh) charged per vehicle:
        energy_to_charge = charging_action * self.chariging_power.tolist() * self.dt

        # get current SOC
        currently_charged = self.battery_capacities.tolist() * self.state[self.locations_upper:self.soc_upper]

        # calculate energy to charge for each vehicle
        energy_to_charge = np.minimum(energy_to_charge, self.battery_capacities.tolist() - currently_charged)

        # can not charge cars that are not at the station
        energy_to_charge[not_chargable] = 0

        return energy_to_charge

    def discharging(self, action, not_chargable):
        """
        Calculates energy to discharge for each car, depending on action and current SOC.

        Parameters
        ----------
        action: numpy ndarray
            Values: 0 for do nothing, 1 for charging, and 2 for discharging
        not_chargable: numpy ndarray
            Boolean for each car, if chargable (at the station) or not (on trip or not implemented)
        Returns
        ----------
        rew: numpy ndarray
            Energy to discharge for each car.
        """

        # filter charging action
        discharging_action = np.zeros_like(action)
        discharging_action[action == 2] = 1

        # energy (in kWh) discharged per vehicle:
        energy_to_discharge = discharging_action * self.chariging_power.tolist() * self.dt

        # can only discharge until battery is empty
        currently_charged = self.battery_capacities.tolist() * self.state[self.locations_upper:self.soc_upper]
        energy_to_discharge = -np.minimum(energy_to_discharge, currently_charged)

        # can only discharge cars that are at the station
        energy_to_discharge[not_chargable] = 0

        return energy_to_discharge

    def update_SOC_charging(self, energy_to_charge):
        """
        Updates state of SOC through charging.

        Parameters
        ----------
        energy_to_charge: numpy ndarray
            Energy to charge for each car.
        """

        # update SOC state
        self.state[self.locations_upper:self.soc_upper] += energy_to_charge / self.battery_capacities

        return

    def update_SOC_discharging(self, energy_to_discharge):
        """
        Updates state of SOC through discharging.

        Parameters
        ----------
        energy_to_discharge: numpy ndarray
            Energy to discharge for each car.
        """

        # update SOC state
        self.state[self.locations_upper:self.soc_upper] += energy_to_discharge / self.battery_capacities

        return

    def compute_monetary_reward(self, energy_to_charge, energy_to_discharge, penalty_counter):
        """
        Computes reward at current timestep.

        Parameters
        ----------
        energy_to_charge: numpy ndarray
            Energy to charge for each car.
        energy_to_discharge: numpy ndarray
            Energy to discharge for each car.
        penalty_counter: int
            Total factor of penalty for timestep.

        Returns
        ----------
        rew: float
            Total reward of given timestamp.
        """

        # find cars that started new trip
        on_trip = self.state[:self.locations_upper] > 1000000
        trip_start = self.state[:self.locations_upper] != self.state_old[:self.locations_upper]
        result = on_trip & trip_start
        on_trip_indices = np.where(result)[0].tolist()

        # variable for summing up reward
        sum_reward_trip = 0

        # iterate over cars which started a new trip
        for trip_start in on_trip_indices:
            reservation_no = self.state[trip_start]

            # no reward for relocations
            if reservation_no >= 30000000:
                pass

            else:
                try:

                    # get duration and distance reward
                    duration_reward = \
                    self.reservations[self.reservations["reservation_no"] == reservation_no]["revenue_duration"].iloc[0]
                    distance_reward = \
                    self.reservations[self.reservations["reservation_no"] == reservation_no]["revenue_distance"].iloc[0]

                    # calculate total reward
                    total_reward_trip = duration_reward + distance_reward

                    # sum up reward of all beginning trips
                    sum_reward_trip += total_reward_trip

                    # identify missing data
                    if duration_reward is None:
                        print("")
                        print("Warning: compute_reward()")
                        print("Trip duration revenue is NULL")
                        print("Reservation_number")
                        print(reservation_no)
                    if distance_reward is None:
                        print("")
                        print("Warning: compute_reward()")
                        print("Trip distance revenue is NULL")
                        print("Reservation_number")
                        print(reservation_no)

                # exception no data found
                except Exception as E:
                    print("")
                    print("Warning: compute_reward()")
                    print(E)
                    print("Reservation_number:")
                    print(reservation_no)
                    print("Observed state:")
                    print(self.state[trip_start])
                    print("Previous state (most likely start station):")
                    print(self.state_old[trip_start])
                    print("Vehicle ID:")
                    print(self.vehicles_id[trip_start])

        # cacluate charging costs
        rew_charging = np.sum(-1 * energy_to_charge * self.electricity_price[self.t - int(
            self.timesteps_since_start / self.episode_len) * self.episode_len + self.timesteps_since_start % self.episode_len])

        # penalty for cancelled reservation, because of not enough charged batteries
        reward_cancellations = -self.cancellation_penalty * penalty_counter

        # reward from V2G contract for timestamp
        v2g_reward = 0
        if self.state[self.v2g_lower:self.v2g_upper] > 0:
            v2g_reward = self.v2g_price_discharging[self.t - int(
            self.timesteps_since_start / self.episode_len) * self.episode_len + self.timesteps_since_start % self.episode_len] * abs(sum(energy_to_discharge))

        if self.state[self.v2g_lower:self.v2g_upper] < 0:
            v2g_reward = self.v2g_price_charging[self.t - int(
            self.timesteps_since_start / self.episode_len) * self.episode_len + self.timesteps_since_start % self.episode_len] * abs(sum(energy_to_charge))

        # penalty if not enough energy discharged for V2G morning event during timestamp (discharging event)
        if self.state[self.v2g_lower:self.v2g_upper] > 0 and (self.t - int(self.timesteps_since_start / self.episode_len) * self.episode_len + self.timesteps_since_start % self.episode_len) < self.episode_len / 2 and abs(sum(energy_to_discharge)) < self.v2g_demand_event[0]:
            v2g_reward -= self.v2g_penalty

        # penalty if not enough energy charged for V2G during timestamp (charging event)
        if self.state[self.v2g_lower:self.v2g_upper] < 0 and sum(energy_to_charge) < abs(self.v2g_demand_event[1]):
            v2g_reward -= self.v2g_penalty

        # penalty if not enough energy discharged for V2G evening event during timestamp (discharging event)
        if self.state[self.v2g_lower:self.v2g_upper] > 0 and (self.t - int(
            self.timesteps_since_start / self.episode_len) * self.episode_len + self.timesteps_since_start % self.episode_len) > self.episode_len / 2 and abs(sum(energy_to_discharge)) < self.v2g_demand_event[2]:
            v2g_reward -= self.v2g_penalty

        # reward from reservations, where customer don't show up (0 km driven distance)
        cancelled_revenue = self.reservations[
            (self.reservations["drive_km"] == 0) & (self.reservations["reservationfrom_time_discrete"] == self.t + 1)][
            "revenue_duration"].sum()

        # penalty for less energy at end of episode compared to beginning
        rew_energy_difference = 0
        if self.t - int(
                self.timesteps_since_start / self.episode_len) * self.episode_len + self.timesteps_since_start % self.episode_len == (
                self.episode_len - 2):
            # difference of total charged energy in the system
            difference_enery = sum(
                self.state[self.locations_upper:self.soc_upper] * self.battery_capacities) - self.energy_beginning

            # positive energy difference gives no penalty/reward
            difference_enery = min(difference_enery, 0)

            # calculate energy difference reward
            rew_energy_difference = difference_enery * self.penalty_per_kwh

        # total reward
        rew = rew_charging + sum_reward_trip + reward_cancellations + rew_energy_difference + cancelled_revenue + v2g_reward

        # save rewards for summary analysis
        self.reward_list.append(rew)
        self.reward_list_trips.append(sum_reward_trip + cancelled_revenue)
        self.reward_list_charging.append(rew_charging + rew_energy_difference)
        self.reward_list_cancellation_penalty.append(reward_cancellations)
        self.reward_list_v2g.append(v2g_reward)

        return rew

    def step(self, action):
        """
        Processes enviroment one timestep further.

        Parameters
        ----------
        action: numpy ndarray
            Values: 0 for do nothing, 1 for charging, and 2 for discharging
        penalty_counter: int
            Total factor of penalty for timestep.

        Returns
        ----------
        self.state: Numpy ndarray
            State of environment: Location, SOC, (next reservation, next reservation duration) for each car, accurance of V2G event in the next timestep.
        rew: float
            Total reward of given timestamp.
        done: Boolean
            True if end of current episode.
        """

        # save current state
        self.state_old = self.state.copy()

        # update state of next reservations (timestamp and duration)
        if self.planned_bookings is True:

            # access precomputed states
            if self.precomputed_bookings is True:
                next_reservation = self.planned_reservations.iloc[:, self.t + 1 - int(
                    self.timesteps_since_start / self.episode_len) * self.episode_len + self.timesteps_since_start % self.episode_len].values
                duration_next_reservation = self.planned_durations.iloc[:, self.t + 1 - int(
                    self.timesteps_since_start / self.episode_len) * self.episode_len + self.timesteps_since_start % self.episode_len].values
                self.state[self.soc_upper: self.reservation_time_upper] = next_reservation
                self.state[self.reservation_time_upper:self.v2g_lower] = duration_next_reservation

            # calculate states
            else:
                next_reservation, duration_next_reservation = self.update_reservation_state(False)
                self.state[self.soc_upper: self.reservation_time_upper] = next_reservation
                self.state[self.reservation_time_upper:self.v2g_lower] = duration_next_reservation

        # SOC needs to be positive or equal to zero
        #assert len(self.state[self.locations_upper:self.soc_upper][
        #              self.state[self.locations_upper:self.soc_upper] < 0]) == 0, "Vehicle SOC is wrong."

        # get current car locations
        car_locations = self.daily_data.iloc[:, self.t + 1 - int(
            self.timesteps_since_start / self.episode_len) * self.episode_len + self.timesteps_since_start % self.episode_len].values

        # update state of car locations
        self.state[:self.locations_upper] = car_locations

        # update locations using saved changes in location through car assingment
        # iterate over cars with a location change through car assingment
        changed_reservations = {}
        for index, change in self.changed_vehicles.copy().items():

            # create local copy of car's change
            change_copy = change.copy()

            # iterate over location changes of car
            for element in change:

                # delete location change if time is over
                if self.t + 1 > element[1]:
                    change_copy.remove(element)

            # update location changes of car
            self.changed_vehicles[index] = change_copy
            change = change_copy

            # delete car's change from dictionary because outdated
            if not change_copy:
                del self.changed_vehicles[index]

            # case if reservation different from inital location change is assignend to car in current step
            elif self.state[index] != change_copy[0][2] and self.state[index] > 1000000 and self.state[
                index] < 30000000 and self.state[index] != self.changed_reservations[index]:

                # get reservation information
                reservations_of_car = self.reservations[self.reservations["reservation_no"] == self.state[index]]

                # case if reservation cannot be found --> original car was assinged to other reservation on previous day
                if len(reservations_of_car) == 0:
                    # print(change)
                    # print(index)
                    # print("number in state")
                    # print(self.state[index])
                    # print("")
                    # print("changed state:")
                    # print(change_copy[0][2])
                    # print("Car ID")
                    # print(self.vehicles_id[index])
                    # print("number in discrete table")
                    # print(self.daily_data.iloc[index, self.t + 1 - int(self.timesteps_since_start / self.episode_len) * self.episode_len + self.timesteps_since_start % self.episode_len])
                    # print("number in state")
                    # print(self.state[index])
                    # print("self changed reservation")
                    # print(self.changed_reservations[index])

                    # update location of car and continue with next car
                    self.state[index] = change_copy[0][0]
                    continue

                # case if original car was assinged to other reservation during earlier timestamp on the same day
                if reservations_of_car["drive_firststart_time_discrete"].iloc[0] < self.t + 1:
                    # update location of car and continue with next car
                    self.state[index] = change_copy[0][0]
                    continue

                # get reservation start location
                car_location = reservations_of_car["start_station_no"].iloc[0]

                # get reservation number
                reservation_no = reservations_of_car["reservation_no"].iloc[0]

                # get reservation duration
                reservation_duration = reservations_of_car["reservation_duration"].iloc[0]

                # get required soc during trip
                soc_needed = reservations_of_car["required_soc"].iloc[0] / 100

                # get duration of trip
                drive_duration = reservations_of_car["drive_duration"].iloc[0]

                # save reservation infromations for later car assingment
                changed_reservations[index] = (
                car_location, soc_needed, reservation_no, reservation_duration, drive_duration)

                # update location of car and continue with next car
                self.state[index] = change_copy[0][0]

                # save re-assingend trip as changed location of car
                self.changed_reservations[index] = self.state[index]

            # no new reservations
            else:

                # update location of car
                self.state[index] = change_copy[0][0]

                # iterate over current changes in reservation
        for index, info in changed_reservations.items():

            # ignore cancellation penalty, since booking after assinging target car to other reservation
            reservation_no_list = []
            for reservation in self.changed_vehicles[index]:
                reservation_no_list.append(reservation[2])

            reservation_no = info[2]
            if reservation_no not in reservation_no_list:

                # ignore cancellation penalty, since booking after assinging target car to other reservation --> customer would book anyways other car if possible
                # required soc will not be subtracted, since the following self.cancel_trips_low_SOC() function will update the SOC of the car
                _ = self.car_assignment(info[0], index, info[1], info[2], info[3], info[4], update_soc=False)

                # add change in location to car
                if index in self.changed_vehicles:
                    self.changed_vehicles[index].insert(1, (info[0], self.t + 1 + info[4], info[2]))
                else:
                    self.changed_vehicles[index] = [(info[0], self.t + 1 + info[4], info[2])]

        # cancel reservations if SOC to low for booked trip, update SOC for cars on trip, and calculate cancellation penalty factor
        penalty_counter = self.cancel_trips_low_SOC()

        # check if car usable for charging or discharging
        not_chargable = (self.state[:self.locations_upper] < 1000) | (self.state[:self.locations_upper] > 6000)

        # calculate energy needed for charging and discharging and update state
        energy_to_charge = self.charging(action, not_chargable)
        energy_to_discharge = None
        self.update_SOC_charging(energy_to_charge)
        if self.state[self.v2g_lower:self.v2g_upper] > 0:
            energy_to_discharge = self.discharging(action, not_chargable)
            self.update_SOC_discharging(energy_to_discharge)

        # set negative SOC to zero (number can get negative in the order of e-18 because of floating number imprecision)
        self.state[self.locations_upper:self.soc_upper] = np.maximum(self.state[self.locations_upper:self.soc_upper], 0)

        # compute reward
        rew = self.compute_monetary_reward(energy_to_charge, energy_to_discharge, penalty_counter)

        # add timestamp to date list
        self.date_list.append(self.daily_data.columns[self.t + 1 - int(
            self.timesteps_since_start / self.episode_len) * self.episode_len + self.timesteps_since_start % self.episode_len])

        if self.v2g is True:
            # check for occurance of v2g event, update V2G event state
            if self.t + 2 - int(
                    self.timesteps_since_start / self.episode_len) * self.episode_len + self.timesteps_since_start % self.episode_len >= self.v2g_discharing_morning_start and self.t + 1 - int(
                    self.timesteps_since_start / self.episode_len) * self.episode_len + self.timesteps_since_start % self.episode_len < self.v2g_discharing_morning_end:
                self.state[self.v2g_lower:self.v2g_upper] = self.v2g_demand_event[0]
       

            elif self.t + 2 - int(
                    self.timesteps_since_start / self.episode_len) * self.episode_len + self.timesteps_since_start % self.episode_len >= self.v2g_charing_noon_start and self.t + 1 - int(
                    self.timesteps_since_start / self.episode_len) * self.episode_len + self.timesteps_since_start % self.episode_len < self.v2g_charing_noon_end:
                self.state[self.v2g_lower:self.v2g_upper] = self.v2g_demand_event[1]

            elif self.t + 2 - int(
                    self.timesteps_since_start / self.episode_len) * self.episode_len + self.timesteps_since_start % self.episode_len >= self.v2g_discharing_evening_start and self.t + 1 - int(
                    self.timesteps_since_start / self.episode_len) * self.episode_len + self.timesteps_since_start % self.episode_len < self.v2g_discharing_evening_end:
                self.state[self.v2g_lower:self.v2g_upper] = self.v2g_demand_event[2]
                

            else:
                self.state[self.v2g_lower:self.v2g_upper] = 0

        # stop if day is over
        done = True if self.t - int(
            self.timesteps_since_start / self.episode_len) * self.episode_len + self.timesteps_since_start % self.episode_len == (
                                   self.episode_len - 2) else False

        # update time state
        self.state[self.v2g_upper:] = self.t + 1

        # update time step
        self.t += 1

        # Create summary plots
        if self.plot_state_histogram:
            self.render_state_histogram()
        if self.plot_state_animation:
            self.render_animation()

        #if self.planned_bookings is True:
        #    # transform to dict:
        #    state = {
        #        'locations_of_vehicles': self.state[:self.locations_upper],
        #        'soc_of_vehicles': self.state[self.locations_upper:self.soc_upper],
        #        'planned_reservations': self.state[self.soc_upper:self.reservation_time_upper],
        #        'planned_reservation_durations': self.state[self.reservation_time_upper:self.v2g_lower],
        #        'v2g_event': self.state[self.v2g_lower:self.v2g_upper],
        #        'current_time': self.state[self.v2g_upper:]
        #    }
        #else:
        #    # transform to dict:
        #    state = {
        #        'locations_of_vehicles': self.state[:self.locations_upper],
        #        'soc_of_vehicles': self.state[self.locations_upper:self.soc_upper],
        #        'v2g_event': self.state[self.v2g_lower:self.v2g_upper],
        #        'current_time': self.state[self.v2g_upper:]
        #    }
#


        return self.state, rew, done, {}

    def render_state_histogram(self):
        """
        Plot current state of environment
        """

        # uncomment for smaller car-sharing networks
        # print(f"-------- State at discrete timestamp {self.t} --------")
        # print(f"-------- State at time {self.daily_data.columns[self.t]} --------")
        # print("vehicle ID:", self.vehicles_id)
        # print("at station:", self.state[: self.nr_vehicles])
        # print("SOC       :", np.around(self.state[self.nr_vehicles :], 2))

        # plot histogram of state
        # get timestamp of current state
        timestamp = self.daily_data.columns[self.t]

        # plot location of vehicles as histogram
        on_trip = (self.state[:self.locations_upper] >= 1000000)
        at_station = (self.state[:self.locations_upper] >= 1000) & (self.state[:self.locations_upper] <= 6000)
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

    def render_animation(self):
        """
        Plot animation of current state of environment for each car.
        """
        soc = self.state[self.locations_upper:self.soc_upper]
        side_size = int(np.sqrt(self.nr_vehicles))
        img_w, img_h = side_size, side_size + 1
        assert img_w * img_h > self.nr_vehicles
        img = np.zeros((img_w * img_h, 4))
        at_station = np.zeros(img_w * img_h)
        at_station[:self.nr_vehicles] = self.state[:self.locations_upper]
        img[:self.nr_vehicles] = np.array([my_cmap(val) for val in soc])
        img[at_station < 0] = np.array([0.8, 0.8, 0.8, 1])
        img[at_station > 10000000] = np.array([0.8, 0.2, 0.1, 1])
        show_soc(img.reshape((img_h, img_w, 4)))

    def episode_summary_statistics(self, nr_ticks):
        """
        Plot episode summary statistics.

        Parameters
        ----------
        nr_ticks: int
            Number of xticks in the plot.
        """
        # calculate data
        date_ = self.date_list
        labels = [s[:10] for s in self.date_list]
        rew_trips = sum(self.reward_list_trips)
        rew_charging = sum(self.reward_list_charging)
        rew_cancellation = sum(self.reward_list_cancellation_penalty)
        rew_v2g = sum(self.reward_list_v2g)
        rew_total = sum(self.reward_list)

        # plot reward over time as lineplot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(date_, self.reward_list)
        ax.axhline(y=0, color='black', linestyle='--')
        plt.xticks(rotation=90)
        xtick_labels = ax.get_xticklabels()
        visible_tick_positions = []
        visible_tick_labels = []
        tick_interval = math.ceil(len(self.date_list) / nr_ticks)
        visible_tick_positions = list(range(0, len(self.date_list), tick_interval))
        visible_tick_labels = [labels[i] for i in visible_tick_positions]
        plt.xticks(visible_tick_positions, visible_tick_labels, rotation=90)
        plt.xlabel("Time", labelpad=15)
        plt.ylabel("Reward (CHF)")
        plt.title("Reward over Time ({} - {})".format(date_[0], date_[-1]))
        plt.subplots_adjust(bottom=0.15)
        plt.show()

        # composition of total reward of episode
        categories = ["Trip Revenue", "Charging Costs", "Cancellation Penalty", "V2G", "Total Reward"]
        values = [rew_trips, rew_charging, rew_cancellation, rew_v2g, rew_total]
        fig, ax = plt.subplots()
        ax.bar(categories, values)
        plt.ylabel("Reward [CHF]")
        plt.title("Episodes Reward ({} - {})".format(date_[0], date_[-1]))
        plt.xticks(rotation=90)
        plt.show()

        # print total reward
        print("Total reward ({} - {}):".format(date_[0], date_[-1]))
        print(rew_total)

    def daily_summary_statistics(self):
        """
        Plot daily summary statistics.
        """
        # calculate data
        date_ = self.date_list[-1][:10]
        rew_trips = sum(self.reward_list_trips[-(self.episode_len - 1):])
        rew_charging = sum(self.reward_list_charging[-(self.episode_len - 1):])
        rew_cancellation = sum(self.reward_list_cancellation_penalty[-(self.episode_len - 1):])
        rew_v2g = sum(self.reward_list_v2g[-(self.episode_len - 1):])
        rew_total = sum(self.reward_list[-(self.episode_len - 1):])

        # plot reward over time as lineplot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot([s[-8:-3] for s in self.date_list[-(self.episode_len - 1):]],
                self.reward_list[-(self.episode_len - 1):])
        plt.xticks(rotation=90)
        xtick_labels = ax.get_xticklabels()
        for i, label in enumerate(xtick_labels):
            if i % 2 != 0:
                label.set_visible(False)
        plt.xlabel("Time")
        plt.ylabel("Reward (CHF)")
        plt.title("Reward over Time ({})".format(date_))
        plt.show()

        # composition of total reward of episode
        categories = ["Trip Revenue", "Charging Costs", "Cancellation Penalty", "V2G", "Total Reward"]
        values = [rew_trips, rew_charging, rew_cancellation, rew_v2g, rew_total]
        fig, ax = plt.subplots()
        ax.bar(categories, values)
        plt.ylabel("Reward [CHF]")
        plt.title("Episodes Reward ({})".format(date_))
        plt.xticks(rotation=90)
        plt.show()

        # print total reward of day
        print("Total reward {}:".format(date_))
        print(rew_total)
