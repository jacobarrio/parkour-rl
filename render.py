import torch
import pygame
from environment import ParkourEnv
from train import PolicyNetwork


def render_trained_agent(model_path="parkour_agent.pt"):
    """Load and watch the agent."""
    
    # Setup env
    env = ParkourEnv(render_mode="human")
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    policy = PolicyNetwork(obs_size, action_size, hidden_size=256).to(device)
    
    try:
        policy.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✓ Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"⚠ Model not found. Running with random policy.")
    
    policy.eval()
    
    # Run episodes
    episode = 0
    running = True
    
    while running:
        obs, info = env.reset()
        episode += 1
        episode_reward = 0
        done = False
        
        print(f"\n=== Episode {episode} ===")
        
        # Initialize rendering
        env.render()
        
        while not done:
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    done = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        done = True
                    elif event.key == pygame.K_r:
                        # Reset episode
                        done = True
            
            if done:
                break
            
            # Get action from policy
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action, _, _, _ = policy.get_action_and_value(obs_tensor)
                action = action.item()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
            # Render
            env.render()
        import time
        time.sleep(0.05)  # Slow down rendering
        print(f"Episode {episode} reward: {episode_reward:.2f}")
        
        if not running:
            break
    
    env.close()
    print("\n✓ Rendering stopped")


if __name__ == "__main__":
    print("Controls:")
    print("  ESC - Quit")
    print("  R   - Reset episode")
    print("\nStarting render...\n")
    render_trained_agent()