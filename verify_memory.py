
import os
import pickle
import torch

def verify_memory():
    """
    Verifies the existence and integrity of AI training data and state files.
    
    This function checks:
    1. Existence of both memory and state files
    2. Memory file has sufficient training examples (>100)
    3. State file contains all required neural network components
    
    Returns:
        bool: True if verification passes, False otherwise
    """
    # Check if both required files exist
    if not os.path.exists('ai_memory.pkl') or not os.path.exists('ai_state.pth'):
        print("No saved state found")
        return False
        
    try:
        # Verify memory file integrity and size
        with open('ai_memory.pkl', 'rb') as f:
            memory = pickle.load(f)
            if len(memory) < 100:  # Ensure sufficient training examples
                print("Insufficient training data")
                return False
            print(f"Memory verified: {len(memory)} experiences")
            
        # Verify state file contains all required components
        checkpoint = torch.load('ai_state.pth')
        required_keys = ['policy_net', 'target_net', 'optimizer', 'epsilon', 'steps']
        if all(key in checkpoint for key in required_keys):
            print("State file verified")
            return True
    except Exception as e:
        # Catch any errors during file reading or verification
        print(f"Verification failed: {e}")
        return False
        
if __name__ == "__main__":
    verify_memory()
