
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from constants import WINDOW_WIDTH, WINDOW_HEIGHT, PLAYER_SIZE, PLAYER_SPEED

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.network(x)

class AIPlayer:
    """
    AI player that learns to play the game using Deep Q-learning with a neural network.
    The game state is normalized for better neural network processing.
    """

    def __init__(self, num_cells=10):
        # State space setup
        self.num_cells = num_cells
        self.cell_width = WINDOW_WIDTH // num_cells
        self.cell_height = WINDOW_HEIGHT // num_cells

        # Define possible actions: (dx, dy) pairs for movement
        self.actions = [(0, 0), (-PLAYER_SPEED, 0), (PLAYER_SPEED, 0),
                       (0, -PLAYER_SPEED), (0, PLAYER_SPEED)]

        # Deep Q-Network setup
        self.input_dim = 4  # player_x, player_y, collectible_x, collectible_y
        self.output_dim = len(self.actions)
        self.policy_net = DQN(self.input_dim, self.output_dim)
        self.target_net = DQN(self.input_dim, self.output_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        # Learning parameters
        self.epsilon = 0.3
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.99
        self.batch_size = 32
        self.target_update = 10
        self.steps = 0

        # Experience replay memory
        self.memory = []
        self.memory_size = 10000

    def get_state(self, player_x, player_y, collectible_x, collectible_y):
        """Normalize the game state for neural network input"""
        state = torch.tensor([
            player_x / WINDOW_WIDTH,
            player_y / WINDOW_HEIGHT,
            collectible_x / WINDOW_WIDTH,
            collectible_y / WINDOW_HEIGHT
        ], dtype=torch.float32)
        return state

    def get_action(self, state):
        """Choose action using epsilon-greedy policy with neural network"""
        if np.random.random() < self.epsilon:
            return np.random.randint(len(self.actions))

        with torch.no_grad():
            q_values = self.policy_net(state)
            return torch.argmax(q_values).item()

    def update(self, old_state, action, reward, new_state):
        """Update neural network using experience replay"""
        # Store experience in memory
        self.memory.append((old_state, action, reward, new_state))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

        # Print learning progress
        if reward != 0:  # Only print when there's a reward
            print("\nAI Update:")
            print(f"State: {old_state}")
            print(f"Action: {action} {self.actions[action]}")
            print(f"Reward: {reward}")
            print(f"New State: {new_state}")

            # Get Q-values for current state
            with torch.no_grad():
                q_values = self.policy_net(old_state)
            
            print("\nQ-Values for Current State:")
            print("+---------+--------+---------+-------+--------+")
            print("| Stay    | Left   | Right   | Up    | Down   |")
            print("+---------+--------+---------+-------+--------+")
            print(f"| {q_values[0]:7.2f} | {q_values[1]:6.2f} | {q_values[2]:7.2f} | {q_values[3]:5.2f} | {q_values[4]:6.2f} |")
            print("+---------+--------+---------+-------+--------+")

        # Perform learning update if enough samples
        if len(self.memory) >= self.batch_size:
            # Sample random batch
            batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
            states = torch.stack([self.memory[i][0] for i in batch])
            actions = torch.tensor([self.memory[i][1] for i in batch])
            rewards = torch.tensor([self.memory[i][2] for i in batch], dtype=torch.float32)
            next_states = torch.stack([self.memory[i][3] for i in batch])

            # Compute Q values
            current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
            next_q_values = self.target_net(next_states).max(1)[0].detach()
            target_q_values = rewards + (self.gamma * next_q_values)

            # Compute loss and update
            loss = self.criterion(current_q_values.squeeze(), target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
