
import os
import pickle
import torch

def verify_memory():
    if not os.path.exists('ai_memory.pkl') or not os.path.exists('ai_state.pth'):
        print("No saved state found")
        return False
        
    try:
        with open('ai_memory.pkl', 'rb') as f:
            memory = pickle.load(f)
            if len(memory) < 100:  # Minimum experiences threshold
                print("Insufficient training data")
                return False
            print(f"Memory verified: {len(memory)} experiences")
            
        checkpoint = torch.load('ai_state.pth')
        required_keys = ['policy_net', 'target_net', 'optimizer', 'epsilon', 'steps']
        if all(key in checkpoint for key in required_keys):
            print("State file verified")
            return True
    except Exception as e:
        print(f"Verification failed: {e}")
        return False
        
if __name__ == "__main__":
    verify_memory()
