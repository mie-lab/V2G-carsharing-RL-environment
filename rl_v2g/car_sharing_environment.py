import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import bisect
import random
import pandas as pd
import math
from rl_v2g.plotting import show_soc, my_cmap


class CarsharingEnv(gym.Env):
    def __init__(self, stations, vehicle_information, daily_data, reservations, electricity_price, timesteps_since_start,
                 planned_bookings=False, precomputed_bookings=True, planned_reservations=None, planned_durations=None,
                 location_state="binary", normalize_state=False, random_seed_number=122, episode_len=24, dt=0.25,
                 last_timestep=55392, max_charging_power=11, soc_initial_low=0.5, soc_initial_high=1,
                 max_distance_car_assignment=1000, penalty_per_kwh=0.0, cancellation_penalty=100, v2g=True,
                 v2g_price=None, v2g_demand_event_min=500, v2g_demand_event_max=500, v2g_max_duration=2.0,
                 v2g_penalty=499, v2g_penalty_quantity_dependent=False, v2g_probability_charging_event=0.5,
                 v2g_probability_discharging_event=0.5, v2g_morning_time_period=[6.0, 9.0, 11.0],
                 v2g_noon_time_period=[11.25, 14.0, 15.75], v2g_evening_time_period=[16.0, 20.0, 22.0],
                 plot_state_histogram=False, plot_state_animation=False):
        """
        Initialization of simulation for car-sharing charging optimization including vehicle-to-grid (V2G) and
        grid-to-vehicle (G2V) operations. The software is compatible with the OpenAI Gym standard and is usable for
        applications using reinforcement learning.

        Parameters
        ----------
        
        # Required information
        stations: Geopandas Geodataframe
            Car-sharing stations data with the features "station_no", "geom", "x", and "y".
        vehicle_information: Pandas Dataframe
            Vehicle data with the features "vehicle_no", "charge_power", "battery_capacity", and "vehicle_category" for
            each car.
        daily_data: Pandas DataFrame
            Contains the historical car-sharing trips over the current day.  
        reservations: Pandas DataFrame
            Includes the features "reservation_no", "start_station_no", "syscreatedate_time_discrete", "vehicle_no",
            "drive_firststart_time_discrete", "drive_lastend_time_discrete", "reservationfrom_time_discrete",
            "revenue_distance", "revenue_duration", "revenue_distance", "drive_km", "drive_duration", "required_soc",
            and "reservation_duration" for each reservation.
        electricity_price: Numpy ndarray
            Contains electricity prices for each discrete timestamp of one day.
        timesteps_since_start: Int
            Timestamp since start of simulation (t=0) before current day.
            
        # Simulation of planned bookings (optional)
        planned_bookings: Boolean, optional
            Boolean indicating whether there are planned bookings in the environment; otherwise, all bookings are
            spontaneous, by default True.
        precomputed_bookings: Boolean, optional
            Boolean indicating whether the planned bookings in the environment are precomputed (for massive speed up),
            by default True.
        planned_reservations: Pandas DataFrame, optional
            Timestamps of next planned reservations for each car. Required if planned_bookings is True. By default None. 
        planned_durations: Pandas DataFrame, optional
            Durations of next planned reservations for each car. Required if planned_bookings is True. by default None.
            
        # Observation space settings (state) (optional)
        location_state: String, optional
            Three options, by default "binary:
            - "station_id": 
                - Station number (1000-6000)
                - Reservation number (2x'xxx'xxx) during trip
                - Reservation number (3x'xxx'xxx) during relocation
                - -1 if car is not available
            - "geocoordinates":
                - Coordinates (x, y) of station in LV95
                - Coordinates far away for cars not at station (1000000, 0)
            - "binary:
                - 0 if car is not available
                - 1 if car is at a station
        normalize_state: Boolean, optional
            Boolean indicating whether to apply min-max normalization (0-1) to state featues, by default False.
            
        # simulation settings (optional)
        random_seed_number: Int, optional
            Used for fixing random components in environment for reproducable results, by default 122.
        episode_len: Int, optional
            Length of one episode in hours, by default 24.
            Attention! Software was not tested using non-default value.
        dt: Float, optional
            Time step size in hours, by default 0.25 (a quarter hour).
            Attention! Software was not tested using non-default value.
        last_timestep: int, optional
            Last discrete timestep of simulation period, by defualt 55392.
        max_charging_power: Int, optional
            Maximum charging power suported at the stations, by default 11.
        soc_initial_low: Float, optional
            Lower bound (0-1) for initial SOC used for the reset of the environment, by default 0.5 (50%).
        soc_initial_high: Float, optional
            Upper bound (0-1) for initial SOC used for the reset of the environment, by default 1.0 (100%).
        max_distance_car_assignment: Int, optional
            Maximum search distance in meter for car-assignment problem, by default 1000.
        penalty_per_kwh: Int, optional
            Penalty in CHF/kWh for negative energy difference (total over fleet) between beginning and ending of episode,
            by default 0.0.
        cancellation_penalty: Int, optional
            Maximum penalty in CHF for a booking cancelled due to not enough charged battery, by default 100.
      
        # Simulation of grid operations (V2G/G2V) (optional)
        v2g: Boolean, optional
            Boolean indicating if V2G events take place, by default True.
        v2g_price: Numpy ndarray, optional
            Numpy array of length episode_len containing prices in CHF per kWh, by default None.
            Required if v2g is True.  
        v2g_demand_event_min: Int, optional
            Minimum energy demand during V2G event in kWh per time step (dt), by default 500.
        v2g_demand_event_max: Int, optional
            Maximum energy demand during V2G event in kWh per time step (dt), by default 500.
        v2g_max_duration: Float, optional
            Maximum duration of V2G charging or discharging event in hours, by default 2.0.
        v2g_penalty: Int, optional
            Penalty in CHF if agent charges/discharges less energy than required during V2G/G2V event, by default 499.
        v2g_penalty_quantity_dependent: Boolean, optional
            Whether the v2g_penalty is constant or dependent on the difference between required and delivered energy,
            by default False.
        v2g_probability_charging_event: Float, optional
            Probability that a charging event will take place around noon, by default 0.5 (50%).
        v2g_probability_discharging_event: float, optional
            Probability that a discharging event will take place in the morning or evening, by default 0.5 (50%).
        v2g_morning_time_period: List, optional
            Time period must end before noon and should not overlap with v2g_noon_time_period and v2g_evening_time_period.
            List containing: 1) first possible time (hours) for starting v2g discharging event in the morning,
                                by default 6.0 (6 AM).
                             2) last possible time (hours) for starting v2g discharging event in the morning, by default
                                9.0 (9 AM).
                             3) last possible timestamp for v2g discharging operation, by default 11.0 (11:00 AM).
        v2g_noon_time_period: List, optional
            Time period  should not overlap with v2g_morning_time_period and v2g_evening_time_period.
            List containing: 1) first possible time (hours) for starting v2g charging event at noon, by default 11.25
                                (11:25 AM).
                             2) last possible time (hours) for starting v2g charging event at noon, by default 14.0
                                (2:00 PM).
                             3) last possible timestamp for v2g charging operation, by default 15.75 (3:45 PM).
        v2g_evening_time_period: list, optional
            Time period must start after noon and should not overlap with v2g_morning_time_period and v2g_noon_time_period.
            List containing: 1) first possible time (hours) for starting v2g discharging event in the evening,
                                by default 16.0 (4:00 PM).
                             2) last possible time (hours) for starting v2g discharging event in the evening, by default
                                20.0 (8:00 PM).
                             3) last possible timestamp for v2g charging operation, by default 22.0 (10:00 PM).
         
        # Visualization (optional)
        plot_state_histogram: Boolean, optional
            Plot current state of environment as histogram or not, by default False.
        plot_state_animation: Boolean, optional
            Plot current state of environment as animation for each car or not, by default False.
        ----------
        Observation space: 5 parts:
        1) Location for each car (three options):
            - "station_id": 
                - Station number (1000-6000)
                - Reservation number (2x'xxx'xxx) during trip
                - Reservation number (3x'xxx'xxx) during relocation
                - -1 if car is not available
            - "geocoordinates":
                - Coordinates (x, y) of station in LV95
                - Coordinates far away for cars not at station (1000000, 0)
            - "binary:
                - 0 if car is not available
                - 1 if car is at a station
        2) State of charge (SOC) for each vehicle (between 0 and 1).
        3) Timesteps to the next planned booking for each car (discrete timesteps (dt) between 0 and 672 (large values
           are rounded down to one week)).
        4) Duration of the next planned booking (discrete timesteps (dt) between 0 and 672 (large values are rounded
           down to one week)).
        5) Variable indicating the demand of a V2G/G2V events in kWh. Negative values refer to G2V, positive values to V2G.
        6) Current time (discrete value indicating the time in self.dt since midnight of the currently simulated day)
        Remark: States 3) and 4) are only included if the variable "planned_bookings" is True.
        ----------
        Action space: 3 actions for each car:
        1) 0 = do nothing.
        2) 1 = charging (G2V and charging in general).
        3) 3 = discharging (V2G).
        """
        
        # observation space (state)
        self.normalize_state = normalize_state
        self.location_state = location_state

        # environment settings
        self.dt = dt
        self.episode_len = int(episode_len / self.dt)
        self.planned_bookings = planned_bookings
        self.precomputed_bookings = precomputed_bookings
        self.max_distance_car_assignment = max_distance_car_assignment
        self.random_seed_number = random_seed_number
        self.last_timestep = last_timestep
        self.soc_initial_low = soc_initial_low
        self.soc_initial_high = soc_initial_high
        self.cancellation_penalty = cancellation_penalty
        self.penalty_per_kwh = penalty_per_kwh

        # get station information
        self.stations = stations

        # get vehicle and charging information
        self.vehicles_id = vehicle_information["vehicle_no"]
        self.nr_vehicles = len(self.vehicles_id)
        self.charging_power = np.full_like(vehicle_information["charge_power"], np.minimum(
                vehicle_information["charge_power"], max_charging_power))
        self.battery_capacities = vehicle_information["battery_capacity"]
        self.vehicle_type = vehicle_information["vehicle_category"]

        # V2G
        self.v2g = v2g
        self.v2g_demand_event_min = v2g_demand_event_min
        self.v2g_demand_event_max = v2g_demand_event_max
        self.v2g_max_duration = v2g_max_duration / self.dt
        self.v2g_penalty = v2g_penalty
        self.v2g_penalty_quantity_dependent = v2g_penalty_quantity_dependent
        self.v2g_probability_charging_event = v2g_probability_charging_event
        self.v2g_probability_discharging_event = v2g_probability_discharging_event
        self.v2g_morning_time_period = v2g_morning_time_period
        self.v2g_noon_time_period = v2g_noon_time_period
        self.v2g_evening_time_period = v2g_evening_time_period

        # Visualizations
        self.plot_state_histogram = plot_state_histogram
        self.plot_state_animation = plot_state_animation

        ### define observation space ###
        observation_space = {}
        space_dim = 0

        
        ## add location to observation space (three options), normalize or not
        # "binary" location state (at station = 1, not at station = 0)
        if self.location_state == "binary":
            observation_space["locations_of_vehicles"] = spaces.Box(low=0, high=1, shape=(self.nr_vehicles,),
                    dtype=np.float64)
            space_dim += self.nr_vehicles

        # "station_id" as location state 
        # station number (1000-5000) or
        # reservation number (2x'xxx'xxx) during trip or
        # reservation number (3x'xxx'xxx) during relocation or
        # -1 if car is not available
        elif self.location_state == "station_id" and self.normalize_state is True:
            observation_space["locations_of_vehicles"] = spaces.Box(low=0, high=1, shape=(self.nr_vehicles,),
                    dtype=np.float64)
            space_dim += self.nr_vehicles
        elif self.location_state == "station_id":
            observation_space["locations_of_vehicles"] = spaces.Box(low=-1, high=31000000, shape=(self.nr_vehicles,),
                    dtype=np.float64)
            space_dim += self.nr_vehicles
            
        # "geocoordinates" as location state
        elif self.location_state == "geocoordinates" and self.normalize_state is True:
            observation_space["locations_of_vehicles_x"] = spaces.Box(low=0, high=1, shape=(self.nr_vehicles,),
                    dtype=np.float64)
            observation_space["locations_of_vehicles_y"] = spaces.Box(low=0, high=1, shape=(self.nr_vehicles,),
                    dtype=np.float64)
            space_dim += self.nr_vehicles * 2
        elif self.location_state == "geocoordinates":
            observation_space["locations_of_vehicles_x"] = spaces.Box(low=1000000, high=2835831,
                    shape=(self.nr_vehicles,), dtype=np.float64)
            observation_space["locations_of_vehicles_y"] = spaces.Box(low=0, high=1299185, shape=(self.nr_vehicles,),
                    dtype=np.float64)
            space_dim += self.nr_vehicles * 2


        ## add vehicle's SOC to observation space
        observation_space["soc_of_vehicles"] = spaces.Box(low=0, high=1, shape=(self.nr_vehicles,), dtype=np.float64)
        space_dim += self.nr_vehicles

        
        ## add vehicle's planned bookings to observation space (optional)
        if self.planned_bookings and self.normalize_state:
            observation_space["planned_reservations"] = spaces.Box(low=0, high=1, shape=(self.nr_vehicles,),
                    dtype=np.float64)
            observation_space["planned_reservation_durations"] = spaces.Box(low=0, high=1, shape=(self.nr_vehicles,),
                    dtype=np.float64)
            space_dim += self.nr_vehicles * 2
        elif self.planned_bookings:
            observation_space["planned_reservations"] = spaces.Box(low=-1, high=672, shape=(self.nr_vehicles,),
                    dtype=np.float64)
            observation_space["planned_reservation_durations"]= spaces.Box(low=-1, high=672, shape=(self.nr_vehicles,),
                    dtype=np.float64)
            space_dim += self.nr_vehicles * 2

            
        ## add V2G events and current time to observation space
        if self.normalize_state:
            observation_space["v2g_event"] = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float64)
            observation_space["current_time"] = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float64)
            space_dim += 2
        else:
            observation_space["v2g_event"] = spaces.Box(low=-self.v2g_demand_event_max, high=self.v2g_demand_event_max,
                    shape=(1,), dtype=np.float64)
            observation_space["current_time"] = spaces.Box(low=0, high=self.last_timestep, shape=(1,), dtype=np.float64)
            space_dim += 2

            
        # Concatenate the low and high values of each component
        low_values = []
        high_values = []
        for component in observation_space.values():
            low_values.append(component.low)
            high_values.append(component.high)
        low = np.concatenate(low_values)
        high = np.concatenate(high_values)

        # Create the final observation space box
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=np.float64,
            shape=(space_dim,)
        )

        # define action space
        self.action_space = spaces.MultiDiscrete([3 for _ in range(self.nr_vehicles)])
        
        
        ## define state boundaries for slicing state array
        # locations upper bound
        self.locations_upper = self.nr_vehicles

        # soc upper bound
        self.soc_upper = 2 * self.nr_vehicles

        if self.planned_bookings is True:

            # planned reservations, upper bound:
            self.reservation_time_upper = self.nr_vehicles * 3

            # v2g, lower bound:
            self.v2g_lower = self.nr_vehicles * 4

            # v2g, upper bound:
            self.v2g_upper = self.nr_vehicles * 4+ 1

            # time, lower bound:
            self.time_lower = self.nr_vehicles * 4

        else:

            # v2g, lower bound:
            self.v2g_lower = self.nr_vehicles * 2

            # v2g, upper bound:
            self.v2g_upper = self.nr_vehicles * 2 + 1

            # time, lower bound:
            self.time_lower = self.nr_vehicles * 2

            
        # load data to simulation
        self.load_new_data(daily_data, reservations, electricity_price, timesteps_since_start, v2g_price,
                planned_reservations, planned_durations)
        
        # reset environment
        self.reset()
            
        
    def load_new_data(self, daily_data, reservations, electricity_price, timesteps_since_start, v2g_price = None,
                      planned_reservations=None, planned_durations=None):
        """
        Loads data to environment for the simulation of a new day.

        Parameters
        ----------
        daily_data: Pandas DataFrame
            Contains the historical car-sharing trips over the current day.  
        reservations: Pandas DataFrame
            Includes the features "reservation_no", "start_station_no", "syscreatedate_time_discrete", "vehicle_no",
            "drive_firststart_time_discrete", "drive_lastend_time_discrete", "reservationfrom_time_discrete",
            "revenue_distance", "revenue_duration", "revenue_distance", "drive_km", "drive_duration", "required_soc",
            and "reservation_duration" for each reservation.
        electricity_price: Numpy ndarray
            Contains electricity prices for each discrete timestamp of one day.
        timesteps_since_start: int
            Timestamp since start of simulation (t=0) before current day.
        v2g_price: Numpy ndarray, optional
            Numpy array of length episode_len containing prices in CHF per kwh. Required if self.v2g is True. By default
            None.
        planned_reservations: Pandas DataFrame, optional
            Timestamps of next planned reservations for each car. Required if self.planned_bookings is True. By default
            None.
        planned_durations: Pandas DataFrame, optional
            Durations of next planned reservations for each car. Required if self.planned_bookings is True. by default
            None.
        """
        
        # load data of new day
        self.reset_daily_data = daily_data
        self.reset_reservations = reservations
        self.reset_electricity_price = electricity_price
        self.reset_timesteps_since_start= timesteps_since_start
        self.reset_v2g_price = v2g_price
        self.reset_planned_reservations = planned_reservations
        self.reset_planned_durations = planned_durations

        
    def normalize_location_x(self, location_x):
        """
        Applies min-max normalization (0-1) to x coordinates in LV95 system.
        
         Parameters
        ----------
        location_x: Numpy ndarray 
            Contains x coordinates for each vehicle in the car-sharing fleet. Vehicles on trip have x = 1000000.

        Returns
        ----------
        Numpy ndarray
            Contains normalized x coordinates for each vehicle in the car-sharing fleet.
        """

        return (location_x -  1000000) / (2817342.7839138974 - 1000000)
    
    
    def normalize_location_y(self, location_y):
        """
        Applies min-max normalization (0-1) to y coordinates in LV95 system.
        
         Parameters
        ----------
        location_y: Numpy ndarray 
            Contains y coordinates for each vehicle in the car-sharing fleet. Vehicles on trip have y = 0.

        Returns
        ----------
        Numpy ndarray
            Contains normalized y coordinates for each vehicle in the car-sharing fleet.
        """
        return (location_y -  0) / (1284313.7693009912 - 0)

    
    def normalize_location_station_id(self, location):
        """
        Applies min-max normalization (0-1) to station_id location. Vehicles on trip are assinged to -1.
        
         Parameters
        ----------
        location: Numpy ndarray 
            Contains station_id location for each vehicle in the car-sharing fleet.

        Returns
        ----------
        Numpy ndarray
            Contains normalized station_id for each vehicle in the car-sharing fleet.
        """

        # set all vehicles currently on trip to encoding -1 (not available). Reservation ID not useful for RL algorithms
        location[location > 6000] = -1
        return (location + 1) / (6000 + 1)

    
    def normalize_planned_reservations(self, planned_reservation):
        """
        Applies min-max normalization (0-1) to planned next reservation state.
        
         Parameters
        ----------
        planned_reservation: Numpy ndarray 
            Contains number of discrete timesteps when the next planned booking takes place.

        Returns
        ----------
        Numpy ndarray
            Contains normalized next planned booking for each vehicle in the car-sharing fleet.
        """

        return (planned_reservation - 0) / (672  - 0)

    
    def normalize_planned_durations(self, planned_durations):
        """
        Applies min-max normalization (0-1) to planned next reservation duration state.
        
         Parameters
        ----------
        planned_durations: Numpy ndarray 
            Contains duration in number of discrete timesteps for the next planned booking of each car.

        Returns
        ----------
        Numpy ndarray
            Contains normalized next planned booking duration for each vehicle in the car-sharing fleet.
        """

        return (planned_durations - 0) / (672 - 0)

    
    def normalize_v2g_events(self, v2g):
        """
        Applies min-max normalization (0-1) to V2G/G2V state.
        
        Parameters
        ----------
        v2g: Numpy ndarray 
            Contains energy demand for next timestep in kWh. Positive value refers to V2G, negative values to G2V.

        Returns
        ----------
        Numpy ndarray
            Contains normalized energy demand for next timestep. Positive value refers to V2G, negative values to G2V.
        """

        return (v2g + self.v2g_demand_event_max) / (self.v2g_demand_event_max + self.v2g_demand_event_max)

    
    def normalize_time(self, time):
        """
        Applies min-max normalization (0-1) to current day time state.
        
        Parameters
        ----------
        time: Numpy ndarray 
            Current time of the day in discrete timesteps.

        Returns
        ----------
        Numpy ndarray
            Contains normalized current time of the day.
        """

        return (time - 0) / (95 - 0)


    def reset(self):
        """
        Reset environment to beginning of new episode. Including initialization of states.

        Returns
        ----------
        state: numpy ndarray
            1) Location for each car (three options):
            - "station_id": 
                - Station number (1000-6000)
                - Reservation number (2x'xxx'xxx) during trip
                - Reservation number (3x'xxx'xxx) during relocation
                - -1 if car is not available
            - "geocoordinates":
                - Coordinates (x, y) of station in LV95
                - Coordinates far away for cars not at station (1000000, 0)
            - "binary:
                - 0 if car is not available
                - 1 if car is at a station
            2) State of charge (SOC) for each vehicle (between 0 and 1).
            3) Timesteps to the next planned booking for each car (discrete timesteps (dt) between 0 and 672 (large
               values are rounded down to one week)).
            4) Duration of the next planned booking (discrete timesteps (dt) between 0 and 672 (large values are rounded
               down to one week)).
            5) Variable indicating the demand of a V2G/G2V events in kWh. Negative values refer to G2V, positive values
               to V2G.
            6) Current time (discrete value indicating the time in self.dt since midnight of the currently simulated day)
            Remark: States 3) and 4) are only included if the variable "planned_bookings" is True.
        """

        # assign location simulation data
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

        # reset planned reservations dict
        self.planned_reservations_car = {key: [] for key in self.vehicles_id}

        # create random V2G events
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

        
        ### initialize state for daytime t = 0 ###
        # 1) car locations
        car_locations = self.daily_data.iloc[:, 0 + self.timesteps_since_start % self.episode_len].values

        # 2) SoC (state of charge) (uniform random values between self.soc_initial_low and self.soc_initial_high)
        rng = np.random.RandomState(self.random_seed_number + self.t)
        car_SOC = rng.uniform(low=self.soc_initial_low, high=self.soc_initial_high, size=self.nr_vehicles)

        # state 5) Binary V2G event
        v2g_event = np.array([0])

        # state 6) current time
        current_time = np.array([self.t - self.timesteps_since_start])

        # create final state, (add planned bookings and durations if needed)
        if self.planned_bookings is True:

            # 3) timestamp of next planned reservation
            # 4) Duration of next planned reservation
            if self.precomputed_bookings is True:

                # load precomputed state
                next_reservation = self.planned_reservations.iloc[:,0 + self.timesteps_since_start %
                        self.episode_len].values
                duration_next_reservation = self.planned_durations.iloc[:,0 + self.timesteps_since_start %
                        self.episode_len].values

            else:

                # calculate state
                next_reservation, duration_next_reservation = self.update_reservation_state(True)

            # concatinate states 1-5
            self.state = np.concatenate(
                [car_locations, car_SOC, next_reservation, duration_next_reservation, v2g_event, current_time])

        else:
            # concatinate states 1-5 without planned boookings (state 3, 4)
            self.state = np.concatenate([car_locations, car_SOC, v2g_event, current_time])
            
        # save total charged energy at beginning of episode (for reward calculation)
        self.energy_beginning = sum(car_SOC * self.battery_capacities)

        # Transform state to desired format (normalize, geocoordinates, binary, station_id)
        state = self.transfrom_state()

        return state
        

    def transfrom_state(self):
        """
        Transforms state to desired format, based on settings.

        Returns
        ----------
        state: numpy ndarray
            1) Location for each car (three options):
            - "station_id": 
                - Station number (1000-6000)
                - Reservation number (2x'xxx'xxx) during trip
                - Reservation number (3x'xxx'xxx) during relocation
                - -1 if car is not available
            - "geocoordinates":
                - Coordinates (x, y) of station in LV95
                - Coordinates far away for cars not at station (1000000, 0)
            - "binary:
                - 0 if car is not available
                - 1 if car is at a station
            2) State of charge (SOC) for each vehicle (between 0 and 1).
            3) Timesteps to the next planned booking for each car (discrete timesteps (dt) between 0 and 672 (large
               values are rounded down to one week)).
            4) Duration of the next planned booking (discrete timesteps (dt) between 0 and 672 (large values are rounded
               down to one week)).
            5) Variable indicating the demand of a V2G/G2V events in kWh. Negative values refer to G2V, positive values
               to V2G.
            6) Current time (discrete value indicating the time in self.dt since midnight of the currently simulated day)
            Remark: States 3) and 4) are only included if the variable "planned_bookings" is True.
            Dependendt on setting normalized or absolute values.
        """
        
        # create copy of state in simulation
        current_state = self.state.copy()

        # create desired state format
        state_elements = []
        space_dim = 0

        
        ## add location to state (three options)
        # location "binary"
        if self.location_state == "binary":
            
            # get station IDs for each vehicle
            location = current_state[:self.locations_upper]
            
            # set locations to 0 for vehicles currently on trip
            location[location > 6000] = 0
            
            # set  locations to 0 for vehicles currently not implemented in system (-1)
            location[location < 0] = 0
            
            # set locations to 1 for vehicles currently at station
            location[location > 0] = 1
            
            # add transformed location to state
            state_elements.append(location)

        # location "station_id"
        elif self.location_state == "station_id":
            
            # get station and reservation ids
            location = current_state[:self.locations_upper]
            
            # normalize if desired
            if self.normalize_state is True:
                location = self.normalize_location_station_id(location)
                
            # add transformed location to state
            state_elements.append(location)

        # location "geocoordinates"
        elif self.location_state == "geocoordinates":
            
            # get coordinates of each  vehicle in system
            coordinates_car = pd.merge(pd.DataFrame(current_state[:self.locations_upper], columns=['station_no']),
                    self.stations, on='station_no', how='left')
            
            # access x and x coordinates
            x_array = coordinates_car['x'].values
            y_array = coordinates_car['y'].values
            
            # assing trips not at station to coordinate (1000000, 0) for having a location for away from stations
            x_array = np.nan_to_num(x_array, nan=1000000)
            y_array = np.nan_to_num(y_array, nan=0)
            x_array = np.squeeze(x_array)
            y_array = np.squeeze(y_array)

            # normalize if desired
            if self.normalize_state is True:
                x_array = self.normalize_location_x(x_array)
                y_array = self.normalize_location_y(y_array)

            # add transformed location to state
            state_elements.append(x_array)
            state_elements.append(y_array)

            
        ## add SoC to state 
        state_elements.append(current_state[self.locations_upper:self.soc_upper])

        
        ## add next planned bookings and durations for each car to state (optional)
        if self.planned_bookings:
            
            # calculate in how many steps the next reservation will take place
            state_next_booking = current_state[self.soc_upper:self.reservation_time_upper] - current_state[
                    self.v2g_upper:] - self.reset_timesteps_since_start
            
            # durations of next planned reservations
            state_duration = current_state[self.reservation_time_upper:self.v2g_lower]
            
            # round bookings further away than one week to one week (those reservations have not really an influence on
            # charging strategy and are outliers)
            state_next_booking[state_next_booking < 0] = 0
            state_next_booking[state_next_booking > 672] = 672
            
            # round bookings longer than one week to one week. The needed SoC is never higher than 1 (100%) and those
            # reservations are outliers.
            state_duration[state_duration < 0] = 0
            state_duration[state_duration > 672] = 672

            # normalize if desired
            if self.normalize_state is True:
                state_next_booking = self.normalize_planned_reservations(state_next_booking)
                state_duration = self.normalize_planned_durations(state_duration)

             # add transformed next planned bookings and durations to state
            state_elements.append(state_next_booking)
            state_elements.append(state_duration)

        
        ## add V2G/G2V variable and current daytime to state
        # normalize if desired
        if self.normalize_state:
            
            # get V2G/G2V demand
            state_v2g = self.normalize_v2g_events(current_state[self.v2g_lower:self.v2g_upper])
            
            # calculate current daytime
            state_time = self.normalize_time(current_state[self.v2g_upper:] - self.reset_timesteps_since_start)
            
        else:
            
             # get V2G/G2V demand
            state_v2g = current_state[self.v2g_lower:self.v2g_upper]
            
            # calculate current daytime
            state_time = current_state[self.v2g_upper:] - self.reset_timesteps_since_start

        # add transformed V2G/G2V variable and day time to state
        state_elements.append(state_v2g)
        state_elements.append(state_time)

        # merge all features to one array
        current_state = np.concatenate(state_elements)
        
        # return observed state
        return current_state


    def next_day(self, daily_data, reservations, electricity_price, timesteps_since_start, v2g_price = None,
                 planned_reservations=None, planned_durations=None):
        """
        Loads data for simulation of new day. Does not reset the enviroment. Allows long term simulation.

        Parameters
        ----------
        daily_data: Pandas DataFrame
            Contains the historical car-sharing trips over the current day.  
       reservations: Pandas DataFrame
            Includes the features "reservation_no", "start_station_no", "syscreatedate_time_discrete", "vehicle_no",
            "drive_firststart_time_discrete", "drive_lastend_time_discrete", "reservationfrom_time_discrete",
            "revenue_distance", "revenue_duration", "revenue_distance", "drive_km", "drive_duration", "required_soc",
            and "reservation_duration" for each reservation.
        electricity_price: Numpy ndarray
            Contains electricity prices for each discrete timestamp of one day.
        timesteps_since_start: Int
            Timestamp since start of simulation (t=0) before current day.
        v2g_price: Numpy ndarray, optional
            Numpy array of length self.episode_len containing prices in CHF per kWh, by default None.
        planned_reservations: Pandas DataFrame, optional
            Timestamps of next planned reservations for each car. Required if planned_bookings is True. By default None. 
        planned_durations: Pandas DataFrame, optional
            Durations of next planned reservations for each car. Required if planned_bookings is True. by default None.
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
        self.reset_timesteps_since_start = timesteps_since_start


    def get_random_v2g_events(self):
        """
        Samples random V2G events during the day. A discharging (V2G) event can take place in the morning and evening,
        a charging event (G2V) at noon.
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

        
        ## sample random v2g event
        
        # morning discharging event:
        if int(random.random() < self.v2g_probability_discharging_event) == 1:
            
            # get random start time
            self.v2g_discharing_morning_start = round(random.uniform(self.v2g_morning_time_period[0],
                    self.v2g_morning_time_period[1]) / 0.25) * 0.25 / self.dt

            # get random end time
            self.v2g_discharing_morning_end = min(self.v2g_morning_time_period[2] / self.dt,
                    self.v2g_discharing_morning_start + round(random.uniform(0.25, self.v2g_max_duration) / 0.25) *
                    0.25 / self.dt)

            # get random energy to deliver in kWh
            self.v2g_demand_event[0] = random.uniform(self.v2g_demand_event_min, self.v2g_demand_event_max)

        # noon charging event
        if int(random.random() < self.v2g_probability_charging_event) == 1:
            
            # get random start time
            self.v2g_charing_noon_start = round(random.uniform(self.v2g_noon_time_period[0],
                    self.v2g_noon_time_period[1]) / 0.25) * 0.25 / self.dt

            # get random end time
            self.v2g_charing_noon_end = min(self.v2g_noon_time_period[2] / self.dt, self.v2g_charing_noon_start +
                    round(random.uniform(0.25, self.v2g_max_duration) / 0.25) * 0.25 / self.dt)
            
            # get random energy to take in kWh
            self.v2g_demand_event[1] = random.uniform(-self.v2g_demand_event_max, -self.v2g_demand_event_min)

        # evening discharging event
        if int(random.random() < self.v2g_probability_discharging_event) == 1:
            # get random start time
            self.v2g_discharing_evening_start = round(random.uniform(self.v2g_evening_time_period[0],
                    self.v2g_evening_time_period[1]) / 0.25) * 0.25 / self.dt

            # get random end time
            self.v2g_discharing_evening_end = min(self.v2g_evening_time_period[2] / self.dt,
                    self.v2g_discharing_evening_start + round(random.uniform(0.25, self.v2g_max_duration) / 0.25) *
                    0.25 / self.dt)

            # get energy to deliver in kwh
            self.v2g_demand_event[2] = random.uniform(self.v2g_demand_event_min, self.v2g_demand_event_max)

        return


    def update_reservation_state(self, reset):
        """
        Cacluates the next planned booking and duration for each vehicle. Only used if self.planned_bookings is True 
        and self.precomputed_bookings is False. Precomputing this functionality leads to massive speed up.
        Parameters
        ----------
        reset: boolean, optional
            Boolean indicating whether to reset the environment or not.

        Returns
        ----------
        next_reservation : numpy ndarray
            Timestamps of next planned reservations for each car. Required if planned_bookings is True. By default None. 
        duration_next_reservation : numpy ndarray
            Durations of next planned reservations for each car. Required if planned_bookings is True. by default None.
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
        Checks if car have enough SOC for serving trip. If not, the reservation will assinged to a alternative available
        car if possible. In those cases, a distant dependent penalty factor will be calculated.
        ----------
        Returns
        ----------
        penalty_counter: int
            Total factor of penalty for current timestep.
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

        # get required SoC values for cars starting trip
        reservations_step = self.reservations.copy()
        soc_needed[trip_indices] = reservations_step.set_index("reservation_no").loc[reservation_nos] \
                ["required_soc"].values / 100

        # calculate difference between SoC of car at the moment and required SoC for the trip
        difference_energy = self.state[self.locations_upper:self.soc_upper] - soc_needed

        # boolean array, True for cars with to low SoC for trip (those reservations will be cancelled/reassinged)
        cancelled_trips = difference_energy < 0

        # booolean array, True for cars without cancelled trip
        not_cancelled_trips = np.logical_not(cancelled_trips)

        # convert to integer array
        not_cancelled_trips_int = not_cancelled_trips.astype(int)

        # calculate new SoC of cars and add result to state (previous timestamps SOC minus SOC needed for trip)
        self.state[self.locations_upper:self.soc_upper] = (self.state[self.locations_upper:self.soc_upper] - soc_needed
                * not_cancelled_trips_int)

        # get indices of cars with cancelled reservation because of low battery level
        indices = np.where(cancelled_trips)[0]

        
        ### assign bookings to other cars if possible ###

        # sum penalty factor of cancellations, used for reward penalty
        penalty_counter = 0

        # iterate over indices of cars with cancelled reservations
        for index in indices:
            found = False
            last_station = 0

            # get reservation information of cancelled booking
            reservations_of_car = self.reservations[self.reservations["reservation_no"] == self.state[index]]

            # if the reservation cannot be found, print warning (mostly wrong data loaded)
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

        # get geodata of station with cancelled reservation
        station = self.stations[self.stations["station_no"] == station_id].copy()

        # create spatial index
        station.sindex

        # if station location not available (if station coordinates of reservation are missing)
        if station.empty:
            
            # return maximum penalty
            penalty = 1
            return penalty

        # get car type of cancelled car
        car_type_cancelled = self.vehicle_type[index_car]

        # query spatial index to get all geometries within distance of self.max_distance_car_assignment
        neighbours = list(self.stations.sindex.query(station["geom"].iloc[0].buffer(self.max_distance_car_assignment),
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
                    if next_reservation_car != -1 and next_reservation_car >= self.t + 1 and next_reservation_car \
                            <= self.t + 1 + reservation_duration:
                        conflicting_reservations = True

                # if alternative car available (no planned, conflicting reservation yet) and SOC high enough for trip
                if conflicting_reservations is False and (self.state[self.locations_upper:self.soc_upper][index] -
                        soc_needed) >= 0 and self.state[:self.locations_upper][index] >= 1000 and self.state[:
                        self.locations_upper][index] <= 1000000 and car_type_cancelled == self.vehicle_type[index]:
                    solution_found = True
                    changed = False

                    # calculate distance weighted penalty (zero if car aviailable at the same station)
                    penalty = row['distance'] / self.max_distance_car_assignment

                    # update future simulation of found car
                    if index in self.changed_vehicles:
                        self.changed_vehicles[index].insert(0, (reservation_no, self.t + 1 + drive_duration,
                                alternative_station_ID))
                    else:
                        self.changed_vehicles[index] = [(reservation_no, self.t + 1 + drive_duration,
                                alternative_station_ID)]

                    if update_soc is True:
                        # remove used SOC during trip from vehicle state
                        self.state[self.locations_upper + index] = (self.state[self.locations_upper + index] - soc_needed)

                    # update state
                    self.state[index] = reservation_no

                    # stopp searching for alternative car if a solution is already found
                    break

            if solution_found is True:
                break

        # return maximum penalty if no alternative found
        if solution_found is False:
            penalty = 1

        # return penalty of cancellation/reassingment
        return penalty

    
    def charging(self, action, not_chargable):
        """
        Calculates energy to charge for each car, depending on action and current SoC.

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
        energy_to_charge = charging_action * self.charging_power.tolist() * self.dt

        # get current SOC
        currently_charged = self.battery_capacities.tolist() * self.state[self.locations_upper:self.soc_upper]

        # calculate energy to charge for each vehicle
        energy_to_charge = np.minimum(energy_to_charge, self.battery_capacities.tolist() - currently_charged)

        # can not charge cars that are not at the station
        energy_to_charge[not_chargable] = 0

        return energy_to_charge

    
    def discharging(self, action, not_chargable):
        """
        Calculates energy to discharge for each car, depending on action and current SoC.

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
        energy_to_discharge = discharging_action * self.charging_power.tolist() * self.dt

        # can only discharge until battery is empty
        currently_charged = self.battery_capacities.tolist() * self.state[self.locations_upper:self.soc_upper]
        energy_to_discharge = -np.minimum(energy_to_discharge, currently_charged)

        # can only discharge cars that are at the station
        energy_to_discharge[not_chargable] = 0

        return energy_to_discharge

    
    def update_SOC_charging(self, energy_to_charge):
        """
        Updates state of SoC through charging.

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
        Updates state of SoC through discharging.

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
        Computes and returns reward at current timestep.

        Parameters
        ----------
        energy_to_charge: numpy ndarray
            Energy to charge for each car.
        energy_to_discharge: numpy ndarray
            Energy to discharge for each car.
        penalty_counter: int
            Total factor of penalty for timestep (cancellation penalty).

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
            
            # get reservation number
            reservation_no = self.state[trip_start]

            # no reward for relocations (trips with reservation number over 30'000'000)
            if reservation_no >= 30000000:
                pass

            else:
                try:

                    # get duration and distance reward
                    duration_reward = self.reservations[self.reservations["reservation_no"] == reservation_no] \
                            ["revenue_duration"].iloc[0]
                    distance_reward = self.reservations[self.reservations["reservation_no"] == reservation_no] \
                            ["revenue_distance"].iloc[0]

                    # calculate total reward of trip
                    total_reward_trip = duration_reward + distance_reward

                    # sum up reward of all beginning trips
                    sum_reward_trip += total_reward_trip

                    # identify missing data (most likely error in data loading)
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

                # exception no data found (most likely error in data loading)
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
        rew_charging = np.sum(-1 * energy_to_charge * self.electricity_price[self.t - int(self.timesteps_since_start /
                self.episode_len) * self.episode_len + self.timesteps_since_start % self.episode_len])

        # penalty for cancelled reservation, because of not enough charged batteries
        reward_cancellations = -self.cancellation_penalty * penalty_counter

        # reward from V2G contract for timestamp
        v2g_reward = 0
        if self.state[self.v2g_lower:self.v2g_upper] > 0:
            # two approaches, one for reinforcement learning and one for Elie's simulations.
            try:
                v2g_reward = self.v2g_price_discharging[self.t - int(self.timesteps_since_start / self.episode_len) *
                        self.episode_len + self.timesteps_since_start % self.episode_len] * min(abs(sum(
                        energy_to_discharge)), abs(self.state[self.v2g_lower:self.v2g_upper]))[0]
            except:
                v2g_reward = self.v2g_price_discharging[self.t - int(self.timesteps_since_start / self.episode_len) *
                        self.episode_len + self.timesteps_since_start % self.episode_len] * min(abs(sum(
                        energy_to_discharge)), abs(self.state[self.v2g_lower:self.v2g_upper]))

        if self.state[self.v2g_lower:self.v2g_upper] < 0:
            try:
                v2g_reward = self.v2g_price_charging[self.t - int(self.timesteps_since_start / self.episode_len) *
                        self.episode_len + self.timesteps_since_start % self.episode_len] * min(abs(sum(
                        energy_to_charge)), abs(self.state[self.v2g_lower:self.v2g_upper]))[0]
            except:
                 v2g_reward = self.v2g_price_charging[self.t - int(self.timesteps_since_start / self.episode_len)
                        * self.episode_len + self.timesteps_since_start % self.episode_len] * min(abs(sum(
                        energy_to_charge)), abs(self.state[self.v2g_lower:self.v2g_upper]))

        # penalty if not enough energy discharged for V2G morning event during timestamp (discharging event)
        if self.state[self.v2g_lower:self.v2g_upper] > 0 and (self.t - int(self.timesteps_since_start /
                self.episode_len) * self.episode_len + self.timesteps_since_start % self.episode_len) < \
                self.episode_len / 2 and abs(sum(energy_to_discharge)) < abs(self.v2g_demand_event[0]):

            if self.v2g_penalty_quantity_dependent:
                factor = (abs(sum(energy_to_discharge)) / self.v2g_demand_event[0])
            else:
                factor = 0
            v2g_reward -= (self.v2g_penalty - self.v2g_penalty * factor)

        # penalty if not enough energy charged for V2G during timestamp (charging event)
        if self.state[self.v2g_lower:self.v2g_upper] < 0 and sum(energy_to_charge) < abs(self.v2g_demand_event[1]):
            if self.v2g_penalty_quantity_dependent:
                factor = (abs(sum(energy_to_charge)) / abs(self.v2g_demand_event[1]))
            else:
                factor = 0
            v2g_reward -= (self.v2g_penalty - self.v2g_penalty * factor)

        # penalty if not enough energy discharged for V2G evening event during timestamp (discharging event)
        if self.state[self.v2g_lower:self.v2g_upper] > 0 and (self.t - int(self.timesteps_since_start / self.episode_len)
                * self.episode_len + self.timesteps_since_start % self.episode_len) > self.episode_len / 2 and \
                abs(sum(energy_to_discharge)) < abs(self.v2g_demand_event[2]):

            if self.v2g_penalty_quantity_dependent:
                factor = (abs(sum(energy_to_discharge)) / self.v2g_demand_event[2])
            else:
                factor = 0
            v2g_reward -= (self.v2g_penalty - self.v2g_penalty * factor)

        # reward from reservations, where customer don't show up (0 km driven distance)
        cancelled_revenue = self.reservations[
            (self.reservations["drive_km"] == 0) & (self.reservations["reservationfrom_time_discrete"] == self.t + 1)][
            "revenue_duration"].sum()

        # penalty for less energy at end of episode compared to beginning
        rew_energy_difference = 0
        if self.t - int(self.timesteps_since_start / self.episode_len) * self.episode_len + self.timesteps_since_start % \
                self.episode_len == (self.episode_len - 2):

            # difference of total charged energy in the system
            difference_enery = sum(self.state[self.locations_upper:self.soc_upper] * self.battery_capacities) - \
                    self.energy_beginning

            # positive energy difference gives no penalty/reward
            difference_enery = min(difference_enery, 0.0)

            # calculate energy difference reward
            rew_energy_difference = difference_enery * self.penalty_per_kwh

        # total reward
        rew = rew_charging + sum_reward_trip + reward_cancellations + rew_energy_difference + cancelled_revenue \
                + v2g_reward

        # save rewards for summary analysis
        self.reward_list.append(rew)
        self.reward_list_trips.append(sum_reward_trip + cancelled_revenue)
        self.reward_list_charging.append(rew_charging + rew_energy_difference)
        self.reward_list_cancellation_penalty.append(reward_cancellations)
        self.reward_list_v2g.append(v2g_reward)
        
        return rew


    def perform_actions(self, action):
        """
        Performs actions for next time step.

        Parameters
        ----------
        action: numpy ndarray
            Values: 0 for do nothing, 1 for charging, and 2 for discharging

        Returns
        ----------
        not_chargable: Numpy ndarray
            Boolean for each car, if chargable (at the station) or not (on trip or not implemented)
        energy_to_discharge: Numpy ndarray
            Energy to charge for each car.
        energy_to_discharge: Numpy ndarray
            Energy to discharge for each car.
        """
        # check if car usable for charging or discharging
        not_chargable = (self.state[:self.locations_upper] < 1000) | (self.state[:self.locations_upper] > 6000)

        # calculate energy to charge
        energy_to_charge = self.charging(action, not_chargable)
        
        # charge vehicles (update state)
        self.update_SOC_charging(energy_to_charge)
        
        # discharge only if V2G event happens
        energy_to_discharge = None
        if self.state[self.v2g_lower:self.v2g_upper] > 0:
            
            # calculate energy to discharge
            energy_to_discharge = self.discharging(action, not_chargable)
            
            # discharge vehicles (update state)
            self.update_SOC_discharging(energy_to_discharge)
        
        # set negative SOC to zero (number can get negative in the order of e-18 because of floating number imprecision)
        self.state[self.locations_upper:self.soc_upper] = np.maximum(self.state[self.locations_upper:self.soc_upper], 0)

        # save current state in another variable
        self.state_old = self.state.copy()
        
        
        return not_chargable, energy_to_charge, energy_to_discharge
    
    
    def update_state_next_planned_bookings(self):
        """
        Updates the state of the next planned bookings and durations.
        """
        
        # access precomputed states
        if self.precomputed_bookings is True:
            next_reservation = self.planned_reservations.iloc[:, self.t + 1 - int(self.timesteps_since_start /
                    self.episode_len) * self.episode_len + self.timesteps_since_start % self.episode_len].values
            duration_next_reservation = self.planned_durations.iloc[:, self.t + 1 - int(
                    self.timesteps_since_start / self.episode_len) * self.episode_len + self.timesteps_since_start %
                    self.episode_len].values
            self.state[self.soc_upper: self.reservation_time_upper] = next_reservation
            self.state[self.reservation_time_upper:self.v2g_lower] = duration_next_reservation

        # calculate states if not precomputed
        else:
            next_reservation, duration_next_reservation = self.update_reservation_state(False)
            self.state[self.soc_upper: self.reservation_time_upper] = next_reservation
            self.state[self.reservation_time_upper:self.v2g_lower] = duration_next_reservation
        
        return

    def update_state_location(self):
        """
        Updates the state of the vehicle's location.
        """
        
        # get current car locations
        car_locations = self.daily_data.iloc[:, self.t + 1 - int(self.timesteps_since_start / self.episode_len) *
                self.episode_len + self.timesteps_since_start % self.episode_len].values

        # update state of car locations
        self.state[:self.locations_upper] = car_locations

        # update locations using saved changes in location through car assignment
        # iterate over cars with a location change through car assignment
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
            elif self.state[index] != change_copy[0][2] and self.state[index] > 1000000 and self.state[index] < 30000000 \
                    and self.state[index] != self.changed_reservations[index]:

                # get reservation information
                reservations_of_car = self.reservations[self.reservations["reservation_no"] == self.state[index]]

                # case if reservation cannot be found --> original car was assinged to other reservation on previous day
                if len(reservations_of_car) == 0:
        
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

                # save reservation infromations for later car assignment
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

            # ignore cancellation penalty, since booking after assinging target car to other reservation (in reality,
            # it wouldn't be possible to book a car that is already booked)
            reservation_no_list = []
            for reservation in self.changed_vehicles[index]:
                reservation_no_list.append(reservation[2])

            reservation_no = info[2]
            
            if reservation_no not in reservation_no_list:
                # ignore cancellation penalty, since booking after assinging target car to other reservation -->
                # customer would book anyways other car if possible
                # required soc will not be subtracted, since the following self.cancel_trips_low_SOC() function will
                # update the SOC of the car
                _ = self.car_assignment(info[0], index, info[1], info[2], info[3], info[4], update_soc=False)

                # add change in location to corresponding car
                if index in self.changed_vehicles:
                    self.changed_vehicles[index].insert(1, (info[0], self.t + 1 + info[4], info[2]))
                else:
                    self.changed_vehicles[index] = [(info[0], self.t + 1 + info[4], info[2])]
    
    
    def update_state_v2g_g2v(self):
        """
        Updates the state V2G/G2V events.
        """
        
        # check for occurance of v2g event, update V2G event state
        if self.t + 2 - int(
                self.timesteps_since_start / self.episode_len) * self.episode_len + self.timesteps_since_start % \
                self.episode_len >= self.v2g_discharing_morning_start and self.t + 1 - int(
                self.timesteps_since_start / self.episode_len) * self.episode_len + self.timesteps_since_start % \
                self.episode_len < self.v2g_discharing_morning_end:
            self.state[self.v2g_lower:self.v2g_upper] = self.v2g_demand_event[0]


        elif self.t + 2 - int(
                self.timesteps_since_start / self.episode_len) * self.episode_len + self.timesteps_since_start % \
                self.episode_len >= self.v2g_charing_noon_start and self.t + 1 - int(
                self.timesteps_since_start / self.episode_len) * self.episode_len + self.timesteps_since_start % \
                self.episode_len < self.v2g_charing_noon_end:
            self.state[self.v2g_lower:self.v2g_upper] = self.v2g_demand_event[1]

        elif self.t + 2 - int(
                self.timesteps_since_start / self.episode_len) * self.episode_len + self.timesteps_since_start % \
                self.episode_len >= self.v2g_discharing_evening_start and self.t + 1 - int(
                self.timesteps_since_start / self.episode_len) * self.episode_len + self.timesteps_since_start % \
                self.episode_len < self.v2g_discharing_evening_end:
            self.state[self.v2g_lower:self.v2g_upper] = self.v2g_demand_event[2]

        else:
            self.state[self.v2g_lower:self.v2g_upper] = 0


    def step(self, action):
        """
        Simulates car-sharing system for one timestep. 

        Parameters
        ----------
        action: numpy ndarray
            Values: 0 for do nothing, 1 for charging, and 2 for discharging

        Returns
        ----------
        state: Numpy ndarray
            State of environment: Location, SOC, (next reservation, next reservation duration) for each car, accurance
            of V2G event in the next timestep, and current day time.
        rew: float
            Total reward of given timestamp.
        done: Boolean
            True if end of current episode.
        """
        
        # execute actions
        not_chargable, energy_to_charge, energy_to_discharge = self.perform_actions(action)
        
        # update state of next planned reservations (timestamp and duration) if desired
        if self.planned_bookings is True:
            self.update_state_next_planned_bookings()
            
        # update state of vehicle's locations
        self.update_state_location()
            
        # cancel reservations if SOC to low for booked trip, update SOC for cars on trip, and calculate cancellation
        # penalty factor
        penalty_counter = self.cancel_trips_low_SOC()
    
        # set negative SOC to zero (number can get negative in the order of e-18 because of floating number imprecision)
        self.state[self.locations_upper:self.soc_upper] = np.maximum(self.state[self.locations_upper:self.soc_upper], 0)
        
        # compute reward
        rew = self.compute_monetary_reward(energy_to_charge, energy_to_discharge, penalty_counter)

        # add timestamp to date list
        self.date_list.append(self.daily_data.columns[self.t + 1 - int(
                self.timesteps_since_start / self.episode_len) * self.episode_len + self.timesteps_since_start %
                self.episode_len])

        # update state of V2G/G2V events if desired
        if self.v2g is True:
            self.update_state_v2g_g2v()

        # stop episode if day is over (application in reinforcement learning)
        done = True if self.t - int(
                self.timesteps_since_start / self.episode_len) * self.episode_len + self.timesteps_since_start % \
                self.episode_len == (self.episode_len - 2) else False

        # update time state
        self.state[self.v2g_upper:] = self.t + 1 - self.timesteps_since_start
        self.t += 1

        # Create summary plots if desired
        if self.plot_state_histogram:
            self.render_state_histogram()
        if self.plot_state_animation:
            self.render_animation()

        # transform state to desired format
        state = self.transfrom_state()

        return state, rew, done, {}

    
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
