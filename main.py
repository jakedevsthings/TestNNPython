# main.py
import pygame
import sys
import os
import argparse

from game_state import GameState, check_collision
from renderer import Renderer
from constants import (WINDOW_WIDTH, WINDOW_HEIGHT, PLAYER_SIZE, OBSTACLE_SIZE,
                       COLLECTIBLE_SIZE, WHITE, BLUE, RED, BLACK, ORANGE,
                       GRID_COLOR, PLAYER_SPEED, FPS, NUM_OBSTACLES)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--debug',
	                    action='store_true',
	                    help='Enable debug features')
	parser.add_argument('--ai', action='store_true', help='Enable AI player')
	args = parser.parse_args()
	print("Starting game...")
	print("Initializing pygame...")
	pygame.init()
	screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
	pygame.display.set_caption("Move the Square Game")

	game_state = GameState(args.debug)
	renderer = Renderer(screen, args.debug)
	clock = pygame.time.Clock()

	ai_player = None
	if args.ai:
		from ai_player import AIPlayer
		ai_player = AIPlayer()

	while True:
		state = None
		action = None
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				sys.exit()

		old_x, old_y = game_state.player_x, game_state.player_y

		if args.ai and ai_player is not None:
			state = ai_player.get_state(game_state.player_x,
			                            game_state.player_y,
			                            game_state.collectible[0],
			                            game_state.collectible[1])
			action = ai_player.get_action(state)
			dx, dy = ai_player.actions[action]
			game_state.player_x += dx
			game_state.player_y += dy
		else:
			keys = pygame.key.get_pressed()
			if keys[pygame.K_LEFT]:
				game_state.player_x -= PLAYER_SPEED
			if keys[pygame.K_RIGHT]:
				game_state.player_x += PLAYER_SPEED
			if keys[pygame.K_UP]:
				game_state.player_y -= PLAYER_SPEED
			if keys[pygame.K_DOWN]:
				game_state.player_y += PLAYER_SPEED

		game_state.player_x = max(
		    0, min(WINDOW_WIDTH - PLAYER_SIZE, game_state.player_x))
		game_state.player_y = max(
		    0, min(WINDOW_HEIGHT - PLAYER_SIZE, game_state.player_y))

		for obstacle in game_state.obstacles:
			if check_collision(game_state.player_x, game_state.player_y,
			                   PLAYER_SIZE, obstacle[0], obstacle[1],
			                   OBSTACLE_SIZE):
				game_state.player_x, game_state.player_y = old_x, old_y
				game_state.flash_timer = game_state.flash_duration

		reward = 0
		if check_collision(game_state.player_x, game_state.player_y,
		                   PLAYER_SIZE, game_state.collectible[0],
		                   game_state.collectible[1], COLLECTIBLE_SIZE):
			reward = 100
			game_state.score += 10
			game_state.obstacles = game_state.generate_obstacles()
			game_state.collectible = game_state.generate_collectible()

		if args.ai and ai_player is not None:
			current_state = ai_player.get_state(game_state.player_x,
			                                    game_state.player_y,
			                                    game_state.collectible[0],
			                                    game_state.collectible[1])
			if any(
			    check_collision(game_state.player_x, game_state.player_y,
			                    PLAYER_SIZE, obs[0], obs[1], OBSTACLE_SIZE)
			    for obs in game_state.obstacles):
				reward = -50

			if state is not None and action is not None:
				ai_player.update(state, action, reward, current_state)
				if reward != 0:
					print(f"Score: {game_state.score}")
					print("-" * 40)
			state = current_state

		if game_state.flash_timer > 0:
			game_state.flash_timer -= 1

		renderer.render(game_state)
		clock.tick(FPS)


if __name__ == "__main__":
	main()
