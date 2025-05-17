import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
from collections import deque
import pickle
import time

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class RLAgent:
    def __init__(self, state_size=8, action_size=8, batch_size=64, 
                 gamma=0.99, epsilon=1.0, epsilon_min=0.01, 
                 epsilon_decay=0.995, learning_rate=0.001,
                 memory_size=10000, save_interval=100):
        self.state_size = state_size  # Number of state parameters
        self.action_size = action_size  # 2^3 = 8 possible actions (combinations of left, right, thrust)
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.update_target_model()
        
        self.episode_count = 0
        self.step_count = 0
        self.save_interval = save_interval
        self.total_rewards = []
        self.start_time = time.time()
        self.training_time = 0
        
        self.load_if_exists()
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        # Convert state to tensor
        state = torch.FloatTensor(state).to(self.device)
        
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            # Random action
            action_idx = random.randrange(self.action_size)
        else:
            # Get action from model
            with torch.no_grad():
                q_values = self.model(state)
                action_idx = torch.argmax(q_values).item()
        
        # Convert action index to binary array [left, right, thrust]
        binary_action = self._index_to_binary(action_idx)
        return binary_action, action_idx
    
    def _binary_to_index(self, binary_action):
        # Convert [left, right, thrust] to index
        return (binary_action[0] * 4) + (binary_action[1] * 2) + binary_action[2]
    
    def _index_to_binary(self, index):
        # Convert index to [left, right, thrust]
        return [
            bool((index >> 2) & 1),  # left
            bool((index >> 1) & 1),  # right
            bool(index & 1)          # thrust
        ]
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample minibatch from memory
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor([data[0] for data in minibatch]).to(self.device)
        actions = torch.LongTensor([data[1] for data in minibatch]).to(self.device)
        rewards = torch.FloatTensor([data[2] for data in minibatch]).to(self.device)
        next_states = torch.FloatTensor([data[3] for data in minibatch]).to(self.device)
        dones = torch.FloatTensor([data[4] for data in minibatch]).to(self.device)
        
        # Current Q values
        curr_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values (using target network)
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
        
        # Target Q values
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Calculate loss
        loss = F.mse_loss(curr_q.squeeze(), target_q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def preprocess_state(self, game_state):
        # Extract relevant features from game state
        agent = game_state["agent"]
        
        state = [
            agent["x"] / 800.0,  # Normalize x position
            agent["y"] / 600.0,  # Normalize y position
            agent["vx"] / 10.0,  # Normalize x velocity
            agent["vy"] / 10.0,  # Normalize y velocity
            agent["angle"] / 360.0,  # Normalize angle
            agent["angularVelocity"] / 2.0,  # Normalize angular velocity
            agent["distanceToGround"] / 600.0,  # Normalize distance to ground
            
            # Distance to landing surface (x-direction)
            (agent["x"] - game_state["landingSurface"]["x"]) / 800.0
        ]
        
        return state
    
    # def calculate_reward(self, game_state, prev_state=None):
    #     # Base reward
    #     reward = 0
        
    #     # Check if game is over
    #     if game_state["gameOver"]:
    #         # Success bonus
    #         if game_state["success"]:
    #             reward += 100
    #         else:
    #             reward -= 100  # Crash penalty
    #         return reward
        
    #     agent = game_state["agent"]
    #     landing_surface = game_state["landingSurface"]
        
    #     # Reward for getting closer to landing surface
    #     x_distance = abs(agent["x"] - (landing_surface["x"] + landing_surface["width"]/2))
    #     x_distance_normalized = x_distance / 800.0
    #     reward -= x_distance_normalized * 0.1
        
    #     # Reward for maintaining appropriate velocity
    #     if abs(agent["vx"]) > 2.0:
    #         reward -= 0.1 * abs(agent["vx"])
    #     if abs(agent["vy"]) > 2.0:
    #         reward -= 0.1 * abs(agent["vy"])
        
    #     # Reward for keeping the lander upright
    #     angle_penalty = abs(agent["angle"]) / 180.0
    #     reward -= angle_penalty * 0.2
        
    #     # Penalize fuel usage (thrust)
    #     if prev_state and "action" in prev_state and prev_state["action"][2]:  # If thrust was used
    #         reward -= 0.01
        
    #     # Bonus for being directly above landing pad and descending slowly
    #     if x_distance < landing_surface["width"] and agent["vy"] > 0 and agent["vy"] < 1.5:
    #         reward += 0.2
        
    #     return reward

    # start penalizing for high angular velocity
    # def calculate_reward(self, game_state, prev_state=None):
    #     # Base reward
    #     reward = 0
        
    #     # Check if game is over
    #     if game_state["gameOver"]:
    #         # Success bonus
    #         if game_state["success"]:
    #             reward += 100
    #         else:
    #             reward -= 100  # Crash penalty
    #         return reward
        
    #     agent = game_state["agent"]
    #     landing_surface = game_state["landingSurface"]
        
    #     # Reward for getting closer to landing surface
    #     x_distance = abs(agent["x"] - (landing_surface["x"] + landing_surface["width"]/2))
    #     x_distance_normalized = x_distance / 800.0
    #     reward -= x_distance_normalized * 0.1
        
    #     # Reward for maintaining appropriate velocity
    #     if abs(agent["vx"]) > 2.0:
    #         reward -= 0.1 * abs(agent["vx"])
    #     if abs(agent["vy"]) > 2.0:
    #         reward -= 0.1 * abs(agent["vy"])
        
    #     # Reward for keeping the lander upright
    #     angle_penalty = abs(agent["angle"]) / 180.0
    #     reward -= angle_penalty * 0.2
        
    #     # Strongly penalize high angular velocity (spinning)
    #     angular_velocity_penalty = abs(agent["angularVelocity"]) * 0.5
    #     reward -= angular_velocity_penalty
        
    #     # Bigger penalty for extreme angular velocity
    #     if abs(agent["angularVelocity"]) > 0.5:
    #         reward -= abs(agent["angularVelocity"]) * 1.0
        
    #     # Penalize fuel usage (thrust)
    #     if prev_state and "action" in prev_state and prev_state["action"][2]:  # If thrust was used
    #         reward -= 0.01
        
    #     # Bonus for being directly above landing pad and descending slowly
    #     if x_distance < landing_surface["width"] and agent["vy"] > 0 and agent["vy"] < 1.5:
    #         reward += 0.2
        
    #     # Extra reward for being stable (low angular velocity) above landing pad
    #     if x_distance < landing_surface["width"] and abs(agent["angularVelocity"]) < 0.1:
    #         reward += 0.2
        
    #     return reward


    # take into account the distance to the landing surface
    def calculate_reward(self, game_state, prev_state=None):
        # Base reward
        reward = 0
        
        # Check if game is over
        if game_state["gameOver"]:
            # Success bonus
            if game_state["success"]:
                reward += 100
            else:
                reward -= 100  # Crash penalty
            return reward
        
        agent = game_state["agent"]
        landing_surface = game_state["landingSurface"]
        
        # Reward for getting closer to landing surface horizontally
        x_distance = abs(agent["x"] - (landing_surface["x"] + landing_surface["width"]/2))
        x_distance_normalized = x_distance / 800.0
        reward -= x_distance_normalized * 0.1
        
        # Factor in distance to ground
        distance_to_ground = agent["distanceToGround"]
        
        # Reward for maintaining appropriate velocity
        if abs(agent["vx"]) > 2.0:
            reward -= 0.1 * abs(agent["vx"])
        
        # Vertical velocity should be proportional to height - allow faster descent when high
        # but require slower descent when close to ground
        safe_vy_threshold = min(2.0, 0.1 + distance_to_ground / 20)  # Allows faster descent at greater heights
        
        if agent["vy"] > safe_vy_threshold:  # Descending too fast
            reward -= 0.2 * (agent["vy"] - safe_vy_threshold)
        
        # Reward for keeping the lander upright
        angle_penalty = abs(agent["angle"]) / 180.0
        reward -= angle_penalty * 0.2
        
        # Strongly penalize high angular velocity (spinning)
        angular_velocity_penalty = abs(agent["angularVelocity"]) * 0.5
        reward -= angular_velocity_penalty
        
        # Bigger penalty for extreme angular velocity
        if abs(agent["angularVelocity"]) > 0.5:
            reward -= abs(agent["angularVelocity"]) * 1.0
        
        # Penalize fuel usage (thrust)
        if prev_state and "action" in prev_state and prev_state["action"][2]:  # If thrust was used
            reward -= 0.01
        
        # Height-based orientation requirements
        # When high up, less strict about orientation
        # When close to ground, being upright is more important
        if distance_to_ground < 30:  # Close to ground
            # More strict angle requirement near ground
            reward -= (angle_penalty * 0.5) * (30 - distance_to_ground) / 30
        
        # Bonus for being directly above landing pad and descending controlled
        if x_distance < landing_surface["width"]:
            # Proportional reward based on alignment with landing pad
            alignment_reward = 0.2 * (1 - x_distance / landing_surface["width"])
            reward += alignment_reward
            
            # Additional reward for correct descent rate when above landing pad
            if agent["vy"] > 0 and agent["vy"] < safe_vy_threshold:
                descent_quality = 1 - (agent["vy"] / safe_vy_threshold)
                reward += 0.2 * descent_quality
        
        # Extra reward for being stable (low angular velocity) above landing pad
        if x_distance < landing_surface["width"] and abs(agent["angularVelocity"]) < 0.1:
            # Increase reward for stability as we get closer to ground
            stability_reward = 0.2 * (1 + (1 - min(1, distance_to_ground / 100)))
            reward += stability_reward
        
        return reward
    
    def train(self, game_state, prev_state=None, prev_action_idx=None):
        self.step_count += 1
        
        # Preprocess the current state
        current_state = self.preprocess_state(game_state)
        
        # Calculate reward
        reward = self.calculate_reward(game_state, prev_state)
        
        # Store transition in memory if we have a previous state and action
        if prev_state is not None and prev_action_idx is not None:
            prev_processed_state = self.preprocess_state(prev_state)
            done = game_state["gameOver"]
            
            self.remember(prev_processed_state, prev_action_idx, reward, current_state, done)
            
            # Train the model
            self.replay()
        
        # Update target model periodically
        if self.step_count % 100 == 0:
            self.update_target_model()
        
        # Check if episode has ended
        if game_state["gameOver"]:
            self.episode_count += 1
            self.total_rewards.append(reward)
            
            # Save model periodically
            if self.episode_count % self.save_interval == 0:
                self.save()
                print(f"Episode: {self.episode_count}, Average Reward: {np.mean(self.total_rewards[-100:]):.2f}, Epsilon: {self.epsilon:.4f}")
                # Save training time
                self.training_time = time.time() - self.start_time
                if self.training_time >= 1200:  # 20 mins in seconds
                    print(f"Training completed after {self.training_time/60:.2f} minutes")
            
        # Get action for the current state
        binary_action, action_idx = self.act(current_state)
        
        return binary_action, action_idx, current_state
    
    def predict(self, game_state):
        # For inference only (no exploration)
        state = self.preprocess_state(game_state)
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            q_values = self.model(state_tensor)
            action_idx = torch.argmax(q_values).item()
        
        binary_action = self._index_to_binary(action_idx)
        return {"type": "action", "action": binary_action}
    
    def save(self):
        model_data = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'total_rewards': self.total_rewards,
            'training_time': self.training_time
        }
        torch.save(model_data, 'lunar_lander_model.pth')
        
        # Also save memory for continuing training
        with open('lunar_lander_memory.pkl', 'wb') as f:
            pickle.dump(list(self.memory), f)
        
        print(f"Model saved at episode {self.episode_count}")
    
    def load_if_exists(self):
        if os.path.isfile('lunar_lander_model.pth'):
            try:
                model_data = torch.load('lunar_lander_model.pth', map_location=self.device)
                self.model.load_state_dict(model_data['model_state'])
                self.target_model.load_state_dict(model_data['model_state'])
                self.optimizer.load_state_dict(model_data['optimizer_state'])
                self.epsilon = model_data['epsilon']
                self.episode_count = model_data['episode_count']
                self.step_count = model_data['step_count']
                if 'total_rewards' in model_data:
                    self.total_rewards = model_data['total_rewards']
                if 'training_time' in model_data:
                    self.training_time = model_data['training_time']
                    self.start_time = time.time() - self.training_time
                
                print(f"Loaded model from episode {self.episode_count}")
                
                # Try to load memory
                if os.path.isfile('lunar_lander_memory.pkl'):
                    with open('lunar_lander_memory.pkl', 'rb') as f:
                        self.memory = deque(pickle.load(f), maxlen=self.memory.maxlen)
                    print(f"Loaded memory with {len(self.memory)} examples")
            except Exception as e:
                print(f"Error loading model: {e}")
                # Initialize fresh
                pass