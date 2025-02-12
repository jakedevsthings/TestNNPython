
import pickle
import torch
import numpy as np

# Load the memory
with open('ai_memory.pkl', 'rb') as f:
    memory = pickle.load(f)

# Analyze memory entries
print(f"Total memories stored: {len(memory)}")

# Look for collision events (negative rewards)
collision_states = [(state, action, reward, new_state) 
                   for state, action, reward, new_state in memory 
                   if reward < 0]

print(f"\nFound {len(collision_states)} collision events")

for i, (state, action, reward, new_state) in enumerate(collision_states):
    print(f"\nCollision {i+1}:")
    print(f"State before collision: {state}")
    print(f"Action taken: {action}")
    print(f"New state after collision: {new_state}")
    
    # Calculate normalized positions
    player_x = state[0] * 600  # WINDOW_WIDTH
    player_y = state[1] * 600  # WINDOW_HEIGHT
    print(f"Approximate position: ({player_x:.1f}, {player_y:.1f})")
