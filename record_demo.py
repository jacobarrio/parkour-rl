import cv2
import numpy as np
import pygame
import torch
import time
import sys
from environment import ParkourEnv
from train import PolicyNetwork

def record_demo(output_file="parkour_demo.mp4", duration_sec=60, fps=60):
    """
    Runs the trained agent and records gameplay to an MP4 file.
    """
    print(f"Initializing recording: {output_file} ({duration_sec}s @ {fps}fps)")

    # Setup environment
    env = ParkourEnv(render_mode="human")
    obs, _ = env.reset()
    
    # Setup Video Writer
    # We need to initialize the screen first to get dimensions, although env sets it to 1200x600
    width = 1200
    height = 600
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Load Agent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    policy = PolicyNetwork(obs_size, action_size, hidden_size=256).to(device)
    model_path = "parkour_agent.pt"
    
    try:
        policy.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✓ Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"⚠ Model not found at {model_path}. Using random weights (will look silly).")
    
    policy.eval()

    # Recording Loop
    total_frames = duration_sec * fps
    frame_count = 0
    
    print("\nRecording started... (Press ESC to stop early)")
    
    # Initialize Pygame screen if not already done by env.render()
    # We call render once to ensure window is open
    env.render()
    
    start_time = time.time()
    
    try:
        while frame_count < total_frames:
            # Handle Pygame events to keep window responsive and allow exit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("\nWindow closed.")
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        print("\nRecording stopped by user.")
                        return

            # Get action
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action, _, _, _ = policy.get_action_and_value(obs_tensor)
                action = action.item()

            # Step environment
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if done:
                obs, _ = env.reset()

            # Render to Pygame window
            env.render()
            
            # Capture frame
            # Get image data from Pygame surface
            # Pygame uses (width, height) but opencv expects (height, width, 3)
            # Also Pygame is RGB, OpenCV is BGR
            surface = pygame.display.get_surface()
            if surface is None:
                break
                
            view = pygame.surfarray.array3d(surface)
            # Transpose from (width, height, 3) to (height, width, 3)
            view = view.transpose([1, 0, 2])
            # Convert RGB to BGR
            frame = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
            
            out.write(frame)
            
            frame_count += 1
            
            # Progress update
            if frame_count % 60 == 0:
                percent = (frame_count / total_frames) * 100
                elapsed = time.time() - start_time
                sys.stdout.write(f"\rProgress: {percent:.1f}% | Frame: {frame_count}/{total_frames} | Time: {elapsed:.1f}s")
                sys.stdout.flush()
                
            # Maintain FPS (optional, but good for watching live)
            # Since we are recording, we might want to run as fast as possible or sync to clock.
            # env.clock.tick(fps) handles the waiting if we want real-time viewing.
            # But for recording, we usually just want to capture frames. 
            # However, the user asked to "Show the pygame window... so I can see it happening"
            # So we should respect the clock to keep it watchable.
            env.clock.tick(fps)

    except KeyboardInterrupt:
        print("\nRecording interrupted.")
    finally:
        print("\n\nFinalizing video...")
        out.release()
        env.close()
        print(f"✓ Saved to {output_file}")

if __name__ == "__main__":
    record_demo()
