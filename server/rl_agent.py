import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class RLAgent:
    def __init__(self, state_dim=9, action_space=[[0, 0, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0]]):
        self.model = DQN(state_dim, len(action_space))
        self.action_space = action_space
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def predict(self, state_dict):
        state = self._preprocess_state(state_dict)
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        action_idx = torch.argmax(q_values).item()
        return self.action_space[action_idx]

    def _preprocess_state(self, state):
        agent = state["agent"]
        return np.array([
            agent["x"] / 1000,
            agent["y"] / 1000,
            agent["vx"],
            agent["vy"],
            agent["angle"],
            agent["angularVelocity"],
            agent["distanceToGround"] / 1000,
            1.0 if state["gameOver"] else 0.0,
            1.0 if state["success"] else 0.0,
        ])
