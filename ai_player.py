
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from constants import WINDOW_WIDTH, WINDOW_HEIGHT, PLAYER_SIZE, PLAYER_SPEED

class DQN(nn.Module):
    """
    Deep Q-Network architecture definition.
    A neural network that approximates Q-values for each possible action.
    
    Architecture:
    - Input layer: Game state (player and collectible positions)
    - Hidden layers: 64 neurons -> 32 neurons with ReLU activation
    - Output layer: Q-values for each possible action
    """
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
    AI agent that learns to play the game using Deep Q-Learning.
    Uses experience replay and target network for stable learning.
    
    Features:
    - Experience replay memory to store and learn from past experiences
    - Separate target network for stable Q-value estimation
    - Epsilon-greedy exploration strategy
    - Periodic target network updates
    """

    def __init__(self, num_cells=10):
        # Grid-based state space configuration
        self.num_cells = num_cells
        self.cell_width = WINDOW_WIDTH // num_cells
        self.cell_height = WINDOW_HEIGHT // num_cells
        
        # Try to load saved state
        try:
            if os.path.exists('ai_state.pth') and os.path.exists('ai_memory.pkl'):
                self.load_state()
            else:
                print("No saved state found, starting fresh")
        except Exception as e:
            print(f"Error loading state: {e}, starting fresh")

        # Action space: Stay, Left, Right, Up, Down
        self.actions = [(0, 0), (-PLAYER_SPEED, 0), (PLAYER_SPEED, 0),
                       (0, -PLAYER_SPEED), (0, PLAYER_SPEED)]

        # Neural network initialization
        self.input_dim = 8  # Normalized (player_x, player_y, collectible_x, collectible_y, dist_left, dist_right, dist_top, dist_bottom)
        self.output_dim = len(self.actions)
        self.policy_net = DQN(self.input_dim, self.output_dim)  # Main network for action selection
        self.target_net = DQN(self.input_dim, self.output_dim)  # Target network for stable learning
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimization setup
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        # Learning hyperparameters
        self.epsilon = 0.3        # Initial exploration rate
        self.epsilon_decay = 0.995  # Rate at which exploration decreases
        self.epsilon_min = 0.01   # Minimum exploration rate
        self.gamma = 0.99         # Discount factor for future rewards
        self.batch_size = 32      # Number of experiences to learn from at once
        self.target_update = 10   # How often to update target network
        self.steps = 0            # Counter for target network updates

        # Experience replay memory setup
        self.memory = []          # Storage for past experiences
        self.memory_size = 10000  # Maximum number of stored experiences

    def get_state(self, player_x, player_y, collectible_x, collectible_y):
        """
        Converts raw game coordinates to normalized state representation.
        Includes distances to walls in each direction.
        
        Args:
            player_x, player_y: Player's position coordinates
            collectible_x, collectible_y: Collectible's position coordinates
            
        Returns:
            torch.Tensor: Normalized state vector between 0 and 1
        """
        # Calculate distances to walls
        dist_left = player_x
        dist_right = WINDOW_WIDTH - (player_x + PLAYER_SIZE)
        dist_top = player_y
        dist_bottom = WINDOW_HEIGHT - (player_y + PLAYER_SIZE)

        state = torch.tensor([
            player_x / WINDOW_WIDTH,
            player_y / WINDOW_HEIGHT,
            collectible_x / WINDOW_WIDTH,
            collectible_y / WINDOW_HEIGHT,
            dist_left / WINDOW_WIDTH,
            dist_right / WINDOW_WIDTH,
            dist_top / WINDOW_HEIGHT,
            dist_bottom / WINDOW_HEIGHT
        ], dtype=torch.float32)
        return state

    def get_action(self, state):
        """
        Selects an action using epsilon-greedy policy.
        
        Args:
            state: Current normalized game state
            
        Returns:
            int: Index of the chosen action
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(len(self.actions))  # Explore: random action

        with torch.no_grad():
            q_values = self.policy_net(state)
            return torch.argmax(q_values).item()  # Exploit: best action

    def update(self, old_state, action, reward, new_state):
        """
        Updates the neural network based on the experienced transition.
        Implements experience replay and target network updates.
        
        Args:
            old_state: State before action
            action: Performed action index
            reward: Received reward
            new_state: State after action
        """
        # Store experience in replay memory
        self.memory.append((old_state, action, reward, new_state))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)  # Remove oldest experience if memory is full

        # Print learning progress for significant events
        if reward != 0:
            print("\nAI Update:")
            print(f"State: {old_state}")
            print(f"Action: {action} {self.actions[action]}")
            print(f"Reward: {reward}")
            print(f"New State: {new_state}")

            # Display current Q-values for debugging
            with torch.no_grad():
                q_values = self.policy_net(old_state)
            
            print("\nQ-Values for Current State:")
            print("+---------+--------+---------+-------+--------+")
            print("| Stay    | Left   | Right   | Up    | Down   |")
            print("+---------+--------+---------+-------+--------+")
            print(f"| {q_values[0]:7.2f} | {q_values[1]:6.2f} | {q_values[2]:7.2f} | {q_values[3]:5.2f} | {q_values[4]:6.2f} |")
            print("+---------+--------+---------+-------+--------+")

        # Perform batch learning if enough experiences are collected
        if len(self.memory) >= self.batch_size:
            # Sample and prepare batch data
            batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
            states = torch.stack([self.memory[i][0] for i in batch])
            actions = torch.tensor([self.memory[i][1] for i in batch])
            rewards = torch.tensor([self.memory[i][2] for i in batch], dtype=torch.float32)
            next_states = torch.stack([self.memory[i][3] for i in batch])

            # Compute target Q-values using target network
            current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
            next_q_values = self.target_net(next_states).max(1)[0].detach()
            target_q_values = rewards + (self.gamma * next_q_values)

            # Update policy network
            loss = self.criterion(current_q_values.squeeze(), target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Periodic target network update
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Save state periodically
        if self.steps % 20 == 0:  # Save every 20 steps
            self.save_state()
            
    def save_state(self):
        """Save neural network weights and replay memory"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, 'ai_state.pth')
        
        with open('ai_memory.pkl', 'wb') as f:
            import pickle
            pickle.dump(self.memory, f)
            
    def load_state(self):
        """Load neural network weights and replay memory"""
        checkpoint = torch.load('ai_state.pth')
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        
        with open('ai_memory.pkl', 'rb') as f:
            import pickle
            self.memory = pickle.load(f)
