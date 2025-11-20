import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from environment import ParkourEnv
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """Actor-Critic network."""
    
    def __init__(self, obs_size, action_size, hidden_size=256):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        self.actor = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        features = self.shared(x)
        return self.actor(features), self.critic(features)
    
    def get_action_and_value(self, x, action=None):
        logits, value = self(x)
        dist = Categorical(logits=logits)
        
        if action is None:
            action = dist.sample()
        
        return action, dist.log_prob(action), dist.entropy(), value.squeeze(-1)
    
    def get_value(self, x):
        _, value = self(x)
        return value.squeeze(-1)


def train():
    """Train the agent using PPO."""
    
    # Hyperparams
    num_envs = 4
    num_steps = 128
    total_timesteps = 500_000
    learning_rate = 3e-4
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    ent_coef = 0.05  # Increased from 0.01 for more exploration
    vf_coef = 0.5
    max_grad_norm = 0.5
    batch_size = 512
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Training on {device}")
    print(f"Parallel envs: {num_envs}")
    print(f"Total timesteps: {total_timesteps:,}\n")
    
    # Setup envs
    envs = [ParkourEnv() for _ in range(num_envs)]
    
    # Init network
    obs_size = envs[0].observation_space.shape[0]
    action_size = envs[0].action_space.n
    policy = PolicyNetwork(obs_size, action_size, hidden_size=256).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate, eps=1e-5)
    
    # Reset all envs
    obs = np.array([env.reset()[0] for env in envs])
    
    global_step = 0
    
    while global_step < total_timesteps:
        # Collect rollout
        rollout_obs = []
        rollout_actions = []
        rollout_logprobs = []
        rollout_rewards = []
        rollout_dones = []
        rollout_values = []
        
        for step in range(num_steps):
            global_step += num_envs
            
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).to(device)
                action, logprob, _, value = policy.get_action_and_value(obs_tensor)
            
            rollout_obs.append(obs.copy())
            rollout_actions.append(action.cpu().numpy())
            rollout_logprobs.append(logprob.cpu().numpy())
            rollout_values.append(value.cpu().numpy())
            
            # Step all environments
            next_obs = []
            rewards = []
            dones = []
            
            for i, env in enumerate(envs):
                obs_i, reward, terminated, truncated, _ = env.step(action[i].item())
                done = terminated or truncated
                
                if done:
                    obs_i, _ = env.reset()
                
                next_obs.append(obs_i)
                rewards.append(reward)
                dones.append(done)
            
            obs = np.array(next_obs)
            rollout_rewards.append(np.array(rewards))
            rollout_dones.append(np.array(dones))
        
        # Convert to numpy arrays
        rollout_obs = np.array(rollout_obs)
        rollout_rewards = np.array(rollout_rewards)
        rollout_dones = np.array(rollout_dones)
        rollout_values = np.array(rollout_values)
        
        # Calculate GAE
        with torch.no_grad():
            next_value = policy.get_value(torch.FloatTensor(obs).to(device)).cpu().numpy()
        
        advantages = np.zeros_like(rollout_rewards)
        lastgaelam = 0
        
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - rollout_dones[t]
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - rollout_dones[t + 1]
                nextvalues = rollout_values[t + 1]
            
            delta = rollout_rewards[t] + gamma * nextvalues * nextnonterminal - rollout_values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        
        returns = advantages + rollout_values
        
        # Flatten batch
        b_obs = torch.FloatTensor(rollout_obs.reshape(-1, obs_size)).to(device)
        b_actions = torch.LongTensor(np.array(rollout_actions).flatten()).to(device)
        b_logprobs = torch.FloatTensor(np.array(rollout_logprobs).flatten()).to(device)
        b_advantages = torch.FloatTensor(advantages.flatten()).to(device)
        b_returns = torch.FloatTensor(returns.flatten()).to(device)
        
        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        # PPO update
        batch_indices = np.arange(len(b_obs))
        np.random.shuffle(batch_indices)
        
        for start in range(0, len(b_obs), batch_size):
            end = start + batch_size
            mb_indices = batch_indices[start:end]
            
            _, newlogprob, entropy, newvalue = policy.get_action_and_value(
                b_obs[mb_indices], b_actions[mb_indices]
            )
            
            logratio = newlogprob - b_logprobs[mb_indices]
            ratio = logratio.exp()
            
            mb_advantages = b_advantages[mb_indices]
            
            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            
            # Value loss
            v_loss = 0.5 * ((newvalue - b_returns[mb_indices]) ** 2).mean()
            
            # Entropy bonus
            entropy_loss = entropy.mean()
            
            # Total loss
            loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
        
        # Logging
        if global_step % 5000 == 0:
            avg_reward = rollout_rewards.mean()
            print(f"Step {global_step:,} | Avg Reward: {avg_reward:.2f} | Loss: {loss.item():.4f}")
    
    # Save model
    torch.save(policy.state_dict(), "parkour_agent.pt")
    print("\n✓ Training complete! Model saved to parkour_agent.pt")
    
    for env in envs:
        env.close()


if __name__ == "__main__":
    train()