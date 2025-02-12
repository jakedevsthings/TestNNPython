import random
from constants import (WINDOW_WIDTH, WINDOW_HEIGHT, PLAYER_SIZE, OBSTACLE_SIZE,
                       COLLECTIBLE_SIZE, NUM_OBSTACLES)


class GameState:

	def __init__(self, debug_mode):
		self.debug_mode = debug_mode
		self.score = 0
		self.flash_duration = 10
		self.flash_timer = 0

		# Player state
		self.player_x = WINDOW_WIDTH // 2 - PLAYER_SIZE // 2
		self.player_y = WINDOW_HEIGHT // 2 - PLAYER_SIZE // 2

		# Game objects
		self.obstacles = self.generate_obstacles()

		# Regenerate obstacles if player would spawn on them
		while any(
		    check_collision(self.player_x, self.player_y, PLAYER_SIZE, obs[0],
		                    obs[1], OBSTACLE_SIZE) for obs in self.obstacles):
			self.obstacles = self.generate_obstacles()

		self.collectible = self.generate_collectible()

	def generate_obstacles(self):
		new_obstacles = []
		max_attempts = 100
		for _ in range(NUM_OBSTACLES):
			attempts = 0
			while attempts < max_attempts:
				x = random.randint(PLAYER_SIZE,
				                   WINDOW_WIDTH - OBSTACLE_SIZE - PLAYER_SIZE)
				y = random.randint(PLAYER_SIZE,
				                   WINDOW_HEIGHT - OBSTACLE_SIZE - PLAYER_SIZE)

				if self.is_position_safe_for_obstacle(x, y, new_obstacles):
					new_obstacles.append([x, y])
					break
				attempts += 1
		if self.debug_mode:
			print(f"Obstacles generated at positions: {new_obstacles}")
			print(f"Player position: ({self.player_x}, {self.player_y})")

		return new_obstacles

	def is_position_safe_for_obstacle(self, x, y, current_obstacles):
		# Create a larger safe zone around player spawn
		spawn_x = WINDOW_WIDTH // 2 - PLAYER_SIZE * 3
		spawn_y = WINDOW_HEIGHT // 2 - PLAYER_SIZE * 3
		spawn_size = PLAYER_SIZE * 6  # Even larger safe spawn area

		# Check if obstacle would collide with spawn area
		if (x < spawn_x + spawn_size and x + OBSTACLE_SIZE > spawn_x and
		    y < spawn_y + spawn_size and y + OBSTACLE_SIZE > spawn_y):
			return False

		# Check minimum distance from other obstacles
		for obs in current_obstacles:
			min_distance = PLAYER_SIZE + OBSTACLE_SIZE + 30
			if (abs(x - obs[0]) < min_distance and
			    abs(y - obs[1]) < min_distance):
				return False

		return True

	def generate_collectible(self):
		while True:
			x = random.randint(0, WINDOW_WIDTH - COLLECTIBLE_SIZE)
			y = random.randint(0, WINDOW_HEIGHT - COLLECTIBLE_SIZE)
			if is_position_safe(x, y, COLLECTIBLE_SIZE, self.obstacles):
				if self.debug_mode:
					print(
					    f"Player position: ({self.player_x}, {self.player_y})")
					print(f"Collectible position: ({x}, {y})")
				return [x, y]


def check_collision(x1, y1, size1, x2, y2, size2):
	return (x1 <= x2 + size2 and x1 + size1 >= x2 and y1 <= y2 + size2
	        and y1 + size1 >= y2)


def is_position_safe(x, y, size, obstacles, ignore_player=False):
	if x < PLAYER_SIZE or x > WINDOW_WIDTH - size - PLAYER_SIZE or \
       y < PLAYER_SIZE or y > WINDOW_HEIGHT - size - PLAYER_SIZE:
		return False

	for obs in obstacles:
		if check_collision(x, y, size, obs[0], obs[1], OBSTACLE_SIZE):
			return False

	return True
