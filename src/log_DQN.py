from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from datetime import datetime
from collections import deque
from config_DQN import *

# Log daily reports: Inventory level, in-transit inventory, remaining demand
STATE_ACTION_REPORT_REAL = [[]]  # Real State
COST_RATIO_HISTORY = []

# Record the cumulative value of each cost component
LOG_TOTAL_COST_COMP = {
    'Holding cost': 0,
    'Process cost': 0,
    'Delivery cost': 0,
    'Order cost': 0,
    'Shortage cost': 0
}

class TensorboardLogger:
    """
    Tensorboard logging utility for Single-Agent DQN training

    Args:
        log_dir (str): Directory to save tensorboard logs
    """

    def __init__(self, log_dir='runs'):
        # Create unique run name with timestamp
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        run_name = f'DQN_run_{current_time}'

        # Create log directory
        self.log_dir = os.path.join(log_dir, run_name)
        self.writer = SummaryWriter(self.log_dir)
        self.loss_window = deque(maxlen=100)

        print(f"Tensorboard logging to {self.log_dir}")
        print("To view training progress, run:")
        print(f"tensorboard --logdir={log_dir}")

    def log_training_info(self, episode, episode_reward, avg_cost, inventory_level, epsilon=None): 
        """
        Log training metrics to TensorBoard

        Args:
            episode (int): Current episode number
            episode_reward (float): Total reward for the episode
            avg_cost (float): Average cost per day for the episode
            inventory_level (float): Inventory level for the single agent
            epsilon (float, optional): Current exploration rate
        """
        # Log episode metrics
        self.writer.add_scalar('Training/Episode_Reward', episode_reward, episode)
        self.writer.add_scalar('Training/Average_Daily_Cost', avg_cost, episode)
        self.writer.add_scalar('Inventory/Level', inventory_level, episode)

        # Log exploration rate if provided
        if epsilon is not None:
            self.writer.add_scalar('Training/Epsilon', epsilon, episode)
    
    def log_evaluation_info(self, episode, total_reward, avg_daily_cost, inventory_level):
        """
        Log evaluation metrics to TensorBoard

        Args:
            episode (int): Current evaluation episode
            total_reward (float): Total reward for the evaluation episode
            avg_daily_cost (float): Average daily cost during evaluation
            inventory_level (float): Inventory level for the single agent
        """
        self.writer.add_scalar('Evaluation/Episode_Reward', total_reward, episode)
        self.writer.add_scalar('Evaluation/Average_Daily_Cost', avg_daily_cost, episode)
        self.writer.add_scalar('Evaluation/Inventory_Level', inventory_level, episode)
    
    def log_hyperparameters(self, config_dict): 
        """
        Log hyperparameters to TensorBoard

        Args:
            config_dict (dict): Dictionary containing hyperparameters
        """
        # Log hyperparameters as text
        hyperparams_text = "\n".join([f"{k}: {v}" for k, v in config_dict.items()])
        self.writer.add_text('Hyperparameters', hyperparams_text)

        # Also log as hparams for experiment comparison interface
        self.writer.add_hparams(config_dict, {'hparam/dummy_metric': 0})
    
    def log_replay_buffer_size(self, episode, buffer_size):
        """
        Log Replay Buffer size to TensorBoard

        Args:
            episode (int): Current episode
            buffer_size (int): Replay Buffer size
        """
        self.writer.add_scalar('Training/Replay_Buffer_Size', buffer_size, episode)

    def log_loss(self, loss, step):
        """Log smoothed loss over 100 steps"""
        self.loss_window.append(loss)  
        smoothed_loss = sum(self.loss_window) / len(self.loss_window)  
        self.writer.add_scalar("Loss/train", smoothed_loss, step)  

    def close(self):
        """Close the TensorBoard writer"""
        self.writer.close()
