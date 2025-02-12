import numpy as np
from constants import WINDOW_WIDTH, WINDOW_HEIGHT, PLAYER_SIZE, PLAYER_SPEED


class AIPlayer:
	"""
	AI player that learns to play the game using Q-learning algorithm.
	The game state is discretized into a grid of cells for manageable state space.
	"""

	def __init__(self, num_cells=10):
		# Divide the game window into a grid of num_cells x num_cells
		self.num_cells = num_cells
		self.cell_width = WINDOW_WIDTH // num_cells
		self.cell_height = WINDOW_HEIGHT // num_cells

		# Define possible actions: (dx, dy) pairs for movement
		# (0,0) = stay, (-speed,0) = left, (speed,0) = right, (0,-speed) = up, (0,speed) = down
		self.actions = [(0, 0), (-PLAYER_SPEED, 0), (PLAYER_SPEED, 0),
		                (0, -PLAYER_SPEED), (0, PLAYER_SPEED)]

		# Q-table maps state-action pairs to expected rewards
		# State is represented as (player_cell_x, player_cell_y, collectible_cell_x, collectible_cell_y)
		self.q_table = {}

		# Q-learning hyperparameters
		self.learning_rate = 0.1  # How much to update Q-values (0 to 1)
		self.discount_factor = 0.95  # How much to value future rewards (0 to 1)
		self.epsilon = 0.1  # Probability of choosing random action (exploration)

	def get_state(self, player_x, player_y, collectible_x, collectible_y):
		"""
		Convert continuous game coordinates to discrete cell coordinates.
		Returns a tuple representing the current game state.
		"""
		px = min(self.num_cells - 1, player_x // self.cell_width)
		py = min(self.num_cells - 1, player_y // self.cell_height)
		cx = min(self.num_cells - 1, collectible_x // self.cell_width)
		cy = min(self.num_cells - 1, collectible_y // self.cell_height)
		return (px, py, cx, cy)

	def get_action(self, state):
		"""
		Choose an action using epsilon-greedy policy:
		- With probability epsilon: choose random action (exploration)
		- Otherwise: choose action with highest Q-value (exploitation)
		"""
		if np.random.random() < self.epsilon:
			return np.random.randint(len(self.actions))

		# Initialize Q-values for new states
		if state not in self.q_table:
			self.q_table[state] = np.zeros(len(self.actions))

		return np.argmax(self.q_table[state])

	def update(self, old_state, action, reward, new_state):
		"""
		Update Q-values using the Q-learning update rule:
		Q(s,a) = (1-α)Q(s,a) + α(r + γ*max(Q(s',a')))
		where:
		- α is learning_rate
		- γ is discount_factor
		- r is reward
		- s' is new_state
		"""
		# Initialize Q-values for new states
		if old_state not in self.q_table:
			self.q_table[old_state] = np.zeros(len(self.actions))
		if new_state not in self.q_table:
			self.q_table[new_state] = np.zeros(len(self.actions))

		# Apply Q-learning update rule
		old_value = self.q_table[old_state][action]
		next_max = np.max(self.q_table[new_state])

		new_value = (1 - self.learning_rate) * old_value + \
                   self.learning_rate * (reward + self.discount_factor * next_max)
		self.q_table[old_state][action] = new_value

		# Print learning progress
		if reward != 0:  # Only print when there's a reward (collision or collection)
			print("\nAI Update:")
			print(f"State: {old_state}")
			print(f"Action: {action} {self.actions[action]}")
			print(f"Reward: {reward}")
			print(f"New State: {new_state}")
			print(f"Q-value changed: {old_value:.2f} -> {new_value:.2f}")
