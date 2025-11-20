from environment import ParkourEnv
import time

env = ParkourEnv(render_mode="human", difficulty=1)
obs, _ = env.reset()

print("Testing jump mechanics...")
print("Agent will: fall → land → jump → land")

for step in range(300):
    # Let agent fall and land first
    if step < 60:
        action = 0  # Do nothing
    # Then jump
    elif step == 61:
        action = 3  # Jump
        print(f"Step {step}: JUMP!")
    # Move right while airborne
    elif 62 <= step < 120:
        action = 2  # Right
    else:
        action = 0
    
    obs, reward, done, truncated, _ = env.step(action)
    env.render()
    
    if step % 20 == 0:
        print(f"Step {step}: Y={obs[1]*600:.1f}, Grounded={bool(obs[6])}, Reward={reward:.2f}")
    
    if done or truncated:
        print(f"Episode ended at step {step}")
        break
    
    time.sleep(0.016)  # 60 FPS

env.close()
