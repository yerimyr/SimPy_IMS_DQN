import numpy as np
import os
import torch
from config_SimPy import *
from config_DQN import *
from log_DQN import *
from environment import *
from singleagent_DQN import *

class GymWrapper:
    """
    Wrapper class to handle the interaction between DQN and Gym environment (Single-Agent)
    
    Args:
        env (gym.Env): Gym environment
        action_dim (int): Dimension of the action space
        state_dim (int): Dimension of the state space
        buffer_size (int): Size of the replay buffer
        batch_size (int): Batch size for training
        lr (float): Learning rate for the network
        gamma (float): Discount factor for future rewards
    """

    def __init__(self, env, action_dim, state_dim, buffer_size, batch_size, lr, gamma):
        self.env = env
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.batch_size = batch_size

        # Initialize DQN Agent
        self.dqn = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            buffer_size=buffer_size,
            lr=lr,
            gamma=gamma
        )

        # Initialize TensorBoard logger
        self.logger = TensorboardLogger()

    def train(self, episodes, eval_interval):
        """
        Train the Single-Agent DQN system using the Gym environment
        
        Args:
            episodes (int): Number of training episodes
            eval_interval (int): Interval for evaluation and printing results
        """
        best_reward = float('-inf')  

        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            epsilon = max(0.1, 1.0 - episode / 500)  # Epsilon decay

            while not done:
                # Select action
                action = self.dqn.select_action(state, epsilon)

                # Execute action in environment
                next_state, reward, done, info = self.env.step(action)

                # Store transition in buffer
                self.dqn.buffer.push(state, action, reward, next_state, done)

                episode_reward += reward
                state = next_state

            # Update DQN network
            if len(self.dqn.buffer) >= self.batch_size:
                self.dqn.update(self.batch_size)

            # Log training information
            avg_cost = -episode_reward / self.env.current_day
            self.logger.log_training_info(
                episode=episode,
                episode_reward=episode_reward,
                avg_cost=avg_cost,
                inventory_level=info['Inventory Level'],
                epsilon=epsilon
            )

            # Save the best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                self.save_model(episode, episode_reward)

            # Print logs at evaluation interval
            if episode % eval_interval == 0:
                print(f"Episode {episode}")
                print(f"Episode Reward: {episode_reward}")
                print(f"Average Cost: {avg_cost}")
                print("Inventory Level:", info['Inventory Level'])
                print("-" * 50)

    def evaluate(self, episodes):
        """
        Evaluate the trained DQN system
        
        Args:
            episodes (int): Number of evaluation episodes
        """
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                # Select action without exploration
                action = self.dqn.select_action(state, epsilon=0)

                # Execute action in environment
                state, reward, done, info = self.env.step(action)
                episode_reward += reward

                # Render environment state
                self.env.render()

            avg_daily_cost = -episode_reward / self.env.current_day

            # Log evaluation information
            self.logger.log_evaluation_info(
                episode=episode,
                total_reward=episode_reward,
                avg_daily_cost=avg_daily_cost,
                inventory_level=info['Inventory Level']
            )

            print(f"Evaluation Episode {episode}")
            print(f"Total Reward: {episode_reward}")
            print(f"Average Daily Cost: {avg_daily_cost}")
            print("-" * 50)

    def save_model(self, episode, reward):
        """
        Save the best trained model
        
        Args:
            episode (int): Current episode number
            reward (float): Current episode reward 
        """
        model_path = os.path.join(
            MODEL_DIR, f"dqn_best_model_episode_{episode}.pt")
        torch.save({
            'episode': episode,
            'best_reward': reward,
            'q_network_state_dict': self.dqn.q_network.state_dict(),
            'target_q_network_state_dict': self.dqn.target_network.state_dict(),
            'optimizer_state_dict': self.dqn.optimizer.state_dict()
        }, model_path)
        print(f"Saved best model with reward {reward} to {model_path}")

    def load_model(self, model_path):
        """
        Load a saved model
        
        Args:
            model_path (str): Path to the saved model
        """
        checkpoint = torch.load(model_path)

        # Load model states
        self.dqn.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.dqn.target_network.load_state_dict(checkpoint['target_q_network_state_dict'])

        # Load optimizer state
        self.dqn.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Loaded model from episode {checkpoint['episode']} with best reward {checkpoint['best_reward']}")

    def __del__(self):
        """ Cleanup method to ensure TensorBoard writer is closed """
        if hasattr(self, 'logger'):
            self.logger.close()
