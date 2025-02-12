import pygame
from constants import (WINDOW_WIDTH, WINDOW_HEIGHT, PLAYER_SIZE, OBSTACLE_SIZE,
                       COLLECTIBLE_SIZE, WHITE, BLUE, RED, BLACK, ORANGE,
                       GRID_COLOR)


class Renderer:

	def __init__(self, screen, debug_mode):
		self.screen = screen
		self.debug_mode = debug_mode
		self.font = pygame.font.Font(None, 36)

	def render(self, game_state):
		self.screen.fill(WHITE)
		self._draw_grid()

		self._draw_obstacles(game_state.obstacles)
		self._draw_collectible(game_state.collectible)
		self._draw_player(game_state)
		self._draw_score(game_state.score)

		if self.debug_mode:
			self._draw_debug_markers(game_state)

		pygame.display.flip()

	def _draw_grid(self):
		grid_size = WINDOW_WIDTH // 10  # 10x10 grid to match AI's reference
		for x in range(0, WINDOW_WIDTH, grid_size):
			pygame.draw.line(self.screen, GRID_COLOR, (x, 0),
			                 (x, WINDOW_HEIGHT))
		for y in range(0, WINDOW_HEIGHT, grid_size):
			pygame.draw.line(self.screen, GRID_COLOR, (0, y),
			                 (WINDOW_WIDTH, y))

	def _draw_obstacles(self, obstacles):
		for i, obstacle in enumerate(obstacles):
			pygame.draw.rect(
			    self.screen, RED,
			    (obstacle[0], obstacle[1], OBSTACLE_SIZE, OBSTACLE_SIZE))
			if self.debug_mode:
				pygame.draw.circle(self.screen, BLACK,
				                   (obstacle[0], obstacle[1]), 3)
				number_text = self.font.render(str(i), True, BLACK)
				text_rect = number_text.get_rect(
				    center=(obstacle[0] + OBSTACLE_SIZE / 2,
				            obstacle[1] + OBSTACLE_SIZE / 2))
				self.screen.blit(number_text, text_rect)

	def _draw_collectible(self, collectible):
		pygame.draw.rect(self.screen, BLACK,
		                 (collectible[0], collectible[1], COLLECTIBLE_SIZE,
		                  COLLECTIBLE_SIZE))

	def _draw_player(self, game_state):
		player_color = ORANGE if game_state.flash_timer > 0 else BLUE
		pygame.draw.rect(self.screen, player_color,
		                 (game_state.player_x, game_state.player_y,
		                  PLAYER_SIZE, PLAYER_SIZE))

	def _draw_score(self, score):
		score_text = self.font.render(f'Score: {score}', True, BLACK)
		self.screen.blit(score_text, (10, 10))

	def _draw_debug_markers(self, game_state):
		pygame.draw.circle(self.screen, (255, 0, 0),
		                   (game_state.player_x, game_state.player_y), 3)
		pygame.draw.circle(
		    self.screen, (0, 255, 0),
		    (game_state.collectible[0], game_state.collectible[1]), 3)
