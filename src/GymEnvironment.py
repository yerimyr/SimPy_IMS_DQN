import gym
from gym import spaces
import numpy as np
from config_SimPy import *
from config_DQN import *
from environment import *
from log_SimPy import *
from log_DQN import *

class InventoryManagementEnv(gym.Env):
    """
    Gym environment for single-agent inventory management system
    Handles the simulation of inventory management with a single procurement agent
    """

    def __init__(self):
        super(InventoryManagementEnv, self).__init__()
        self.scenario = {"DEMAND": DEMAND_SCENARIO, "LEADTIME": LEADTIME_SCENARIO}
        self.shortages = 0
        self.total_reward_over_episode = []
        self.total_reward = 0
        self.current_day = 0  # Initialize the current day

        # Record the cumulative value of each cost
        for key in DAILY_COST.keys():
            LOG_TOTAL_COST_COMP[key] = 0

        # Define action space
        """
        Action space is a Discrete space of size ACTION_MAX-ACTION_MIN+1
        """
        self.action_space = spaces.Discrete(ACTION_MAX - ACTION_MIN + 1)

        # Define observation space
        obs_dims = []
        # On-hand inventory level
        for _ in range(len(I)):
            obs_dims.append(INVEN_LEVEL_MAX - INVEN_LEVEL_MIN + 1)
        # In-transition inventory level
        for _ in range(MAT_COUNT):
            obs_dims.append(INVEN_LEVEL_MAX - INVEN_LEVEL_MIN + 1)
        # Remaining demand
        obs_dims.append(INVEN_LEVEL_MAX - INVEN_LEVEL_MIN + 1)

        # Define observation space as MultiDiscrete (for single agent)
        self.observation_space = spaces.MultiDiscrete(obs_dims)

        # Initialize simulation environment
        self.reset()

    def reset(self):
        """
        Reset the environment to initial state

        Returns:
            states: Initial state array
        """
        # Initialize the total reward for the episode
        for key in DAILY_COST.keys():
            LOG_TOTAL_COST_COMP[key] = 0

        # Create new SimPy environment and components
        self.simpy_env, self.inventory_list, self.procurement_list, self.production_list, \
            self.sales, self.customer, self.supplier_list, self.daily_events = create_env(
                I, P, LOG_DAILY_EVENTS
            )

        # Initialize simulation processes
        scenario = {
            "DEMAND": DEMAND_SCENARIO,
            "LEADTIME": LEADTIME_SCENARIO
        }
        simpy_event_processes(
            self.simpy_env, self.inventory_list, self.procurement_list,
            self.production_list, self.sales, self.customer, self.supplier_list,
            self.daily_events, I, scenario
        )
        update_daily_report(self.inventory_list)

        self.current_day = 0
        self.total_reward = 0
        self.shortages = 0

        return self._get_observation()

    def step(self, action):
        """
        Execute one time step (1 day) in the environment

        Args:
            action: Order quantity for the single procurement agent

        Returns:
            observation: State array
            reward: Negative total cost for the day
            done: Whether the episode has ended
            info: Additional information for debugging
        """
        # Set order quantity for the procurement agent
        I[self.procurement_list[0].item_id]["LOT_SIZE_ORDER"] = int(action)

        # Run simulation for one day
        STATE_ACTION_REPORT_REAL[-1].append(action)
        self.simpy_env.run(until=(self.current_day + 1) * 24)
        self.current_day += 1
        update_daily_report(self.inventory_list)

        # Get next observation
        next_state = self._get_observation()

        # Calculate reward (a negative value of the daily total cost)
        reward = -Cost.update_cost_log(self.inventory_list) / 1000  # reward scaling
        # Update LOG_TOTAL_COST_COMP
        for key in DAILY_COST.keys():
            LOG_TOTAL_COST_COMP[key] += DAILY_COST[key]
        Cost.clear_cost()

        self.total_reward += reward
        self.shortages += self.sales.num_shortages
        self.sales.num_shortages = 0

        # Check if episode is done
        done = self.current_day >= SIM_TIME

        # Additional info for debugging
        info = {
            'Day': self.current_day,
            'Daily cost': -reward,
            'Total cost': -self.total_reward,
            'Inventory Level': self.inventory_list[0].on_hand_inventory,
            'In Transit': self.inventory_list[0].in_transition_inventory
        }

        return next_state, reward, done, info

    def _get_observation(self):
        """
        Construct state observation array

        Returns:
            numpy array with shape [STATE_DIM]
        """
        # Initialize single state array
        state = np.zeros(STATE_DIM, dtype=np.int32)  # np.zeros: 모든 값을 0으로 초기화
        state_idx = 0  # 상태 배열의 특정 인덱스를 추적하는 데 사용

        # Add on-hand inventory levels for all items
        for inv in self.inventory_list:
            state[state_idx] = np.clip(
                inv.on_hand_inventory,
                INVEN_LEVEL_MIN,
                INVEN_LEVEL_MAX
            )  # np.clip(value, min, max): value < min -> min반환 / value > max -> max반환 / 그 외는 value 반환
            state_idx += 1

        # Add in-transit inventory levels for material items
        for inv in self.inventory_list:
            if I[inv.item_id]["TYPE"] == "Material":
                state[state_idx] = np.clip(
                    inv.in_transition_inventory,
                    INVEN_LEVEL_MIN,
                    INVEN_LEVEL_MAX  # (Environment.py, Class Procurement)
                )
                state_idx += 1

        # Add remaining demand      
        remaining_demand = I[0]['DEMAND_QUANTITY'] - \
            self.inventory_list[0].on_hand_inventory
        state[state_idx] = np.clip(
            remaining_demand,
            0,
            INVEN_LEVEL_MAX  # if demand > INVEN_LEVEL_MAX: return INVEN_LEVEL_MAX (Config.SimPy.py, def DEMAND_QTY_FUNC())
        )

        return state

    def render(self, mode='human'):
        """
        Render the environment's current state
        Currently just prints basic information
        """
        if mode == 'human':
            print(f"\nDay: {self.current_day}")
            print("\nInventory Levels:")
            print(f"{I[self.inventory_list[0].item_id]['NAME']}: {self.inventory_list[0].on_hand_inventory} "
                  f"(In Transit: {self.inventory_list[0].in_transition_inventory})")

    def close(self):
        """Clean up environment resources"""
        pass
