import numpy as np
import pymunk
import pygame
import gymnasium as gym
from gymnasium import spaces


class ParkourEnv(gym.Env):
    """Simple parkour env using Pymunk for physics."""
    
    metadata = {"render_modes": ["human"], "render_fps": 60}
    
    def __init__(self, render_mode=None, difficulty=1):
        super().__init__()
        
        self.render_mode = render_mode
        self.difficulty = difficulty  # Difficulty settings (1-5)
        
        # Rendering
        self.screen_width = 1200
        self.screen_height = 600
        self.screen = None
        self.clock = None
        
        # Physics
        self.space = pymunk.Space()
        self.space.gravity = (0, 981)  # Downward gravity
        
        # Actions: 0=none, 1=left, 2=right, 3=jump
        self.action_space = spaces.Discrete(4)
        
        # 9D obs: pos, vel, angle, grounded, gap info
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )
        
        # Agent
        self.agent_body = None
        self.agent_shape = None
        
        # Level
        self.ground_bodies = []
        self.platform_edges = []  # Track platform boundaries for observations
        
        # Episode tracking
        self.max_steps = 2000
        self.current_step = 0
        self.start_x = 100
        self.last_x = 100
        self.max_x_reached = 100  # Track furthest progress
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Clear physics
        for body in list(self.space.bodies):
            self.space.remove(body)
        for shape in list(self.space.shapes):
            self.space.remove(shape)
        
        self.ground_bodies = []
        self.platform_edges = []
        self.current_step = 0
        self.last_x = self.start_x
        self.max_x_reached = self.start_x
        
        # Create agent (dynamic box)
        mass = 1
        size = (30, 30)
        moment = pymunk.moment_for_box(mass, size)
        self.agent_body = pymunk.Body(mass, moment)
        self.agent_body.position = (self.start_x, 100)  # Spawn high, let it fall
        self.agent_shape = pymunk.Poly.create_box(self.agent_body, size)
        self.agent_shape.friction = 0.7
        self.space.add(self.agent_body, self.agent_shape)
        
        # Create procedural level
        self._build_procedural_level()
        
        obs = self._get_obs()
        return obs, {}
    
    def _build_procedural_level(self):
        """Generate curriculum-based level with multiple platforms."""
        ground_y = self.screen_height - 50
        
        # Curriculum parameters
        if self.difficulty == 1:
            # Lvl 1: tiny gaps (5-15px)
            num_platforms = 8
            min_gap = 5
            max_gap = 15
            platform_width = 150
        elif self.difficulty == 2:
            # Lvl 2: small jumps (15-40px)
            num_platforms = 10
            min_gap = 15
            max_gap = 40
            platform_width = 120
        elif self.difficulty == 3:
            # Lvl 3: real jumps (40-80px)
            num_platforms = 12
            min_gap = 40
            max_gap = 80
            platform_width = 100
        elif self.difficulty == 4:
            # Lvl 4: big gaps (80-120px)
            num_platforms = 12
            min_gap = 80
            max_gap = 120
            platform_width = 90
        else:  # difficulty == 5
            # Lvl 5: extreme
            num_platforms = 15
            min_gap = 100
            max_gap = 150
            platform_width = 80
        
        # Generate platforms
        current_x = 50
        
        for i in range(num_platforms):
            # Platform segment
            x_start = current_x
            x_end = current_x + platform_width
            self._add_ground_segment(x_start, x_end, ground_y)
            
            # Track edges for observations
            self.platform_edges.append(x_end)
            
            # Add gap (except after last platform)
            if i < num_platforms - 1:
                gap_size = np.random.randint(min_gap, max_gap + 1)
                current_x = x_end + gap_size
                self.platform_edges.append(current_x)
            else:
                current_x = x_end
        
        # Add final long platform (goal area)
        self._add_ground_segment(current_x, current_x + 300, ground_y)
        self.platform_edges.append(current_x + 300)
    
    def _add_ground_segment(self, x1, x2, y):
        """Add static ground segment."""
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = ((x1 + x2) / 2, y)
        shape = pymunk.Segment(body, (x1 - (x1 + x2) / 2, 0), (x2 - (x1 + x2) / 2, 0), 5)
        shape.friction = 1.0
        self.space.add(body, shape)
        self.ground_bodies.append((body, shape, x1, x2))
    
    def step(self, action):
        self.current_step += 1
        
        # Apply action
        if action == 1:  # Left
            self.agent_body.apply_force_at_local_point((-3000, 0), (0, 0))
        elif action == 2:  # Right
            self.agent_body.apply_force_at_local_point((3000, 0), (0, 0))
        elif action == 3:  # Jump (only if grounded)
            if self._is_grounded():
                # Jump impulse: -600 gives ~100px jump height
                self.agent_body.apply_impulse_at_local_point((0, -600), (0, 0))
        
        # Step physics
        self.space.step(1/60.0)
        
        # Cap horizontal velocity
        max_vx = 250
        current_vx = self.agent_body.velocity.x
        if abs(current_vx) > max_vx:
            self.agent_body.velocity = (
                max_vx if current_vx > 0 else -max_vx,
                self.agent_body.velocity.y
            )
        
        # Observation
        obs = self._get_obs()
        
        # Reward shaping
        current_x = self.agent_body.position.x
        
        # 1. Progress reward (only for NEW max distance)
        if current_x > self.max_x_reached:
            progress_reward = (current_x - self.max_x_reached) * 2.0
            self.max_x_reached = current_x
        else:
            progress_reward = 0.0
        
        # 2. Alive bonus
        alive_bonus = 0.1
        
        # 3. Forward velocity bonus (encourage momentum)
        velocity_bonus = max(0, self.agent_body.velocity.x / 100) * 0.5
        
        reward = progress_reward + alive_bonus + velocity_bonus
        
        # Termination conditions
        terminated = False
        truncated = False
        
        # FIXED: Die if fall below screen (not at ground level)
        if self.agent_body.position.y > self.screen_height + 100:
            terminated = True
            reward -= 10  # Death penalty
        
        # Success: reached far right side
        if current_x > self.platform_edges[-1] - 50:
            terminated = True
            reward += 100  # Big success bonus
        
        # Truncate at max steps
        if self.current_step >= self.max_steps:
            truncated = True
        
        self.last_x = current_x
        
        return obs, reward, terminated, truncated, {}
    
    def _get_obs(self):
        """Return agent state with proximity sensors."""
        pos = self.agent_body.position
        vel = self.agent_body.velocity
        
        # Find nearest edge ahead and distance to next platform
        dist_to_edge = 999
        dist_to_next_platform = 999
        
        for i in range(0, len(self.platform_edges) - 1, 2):
            platform_end = self.platform_edges[i]
            next_platform_start = self.platform_edges[i + 1] if i + 1 < len(self.platform_edges) else 999
            
            # If we haven't passed this gap yet
            if platform_end > pos.x:
                dist_to_edge = platform_end - pos.x
                dist_to_next_platform = next_platform_start - pos.x
                break
        
        return np.array([
            pos.x / self.screen_width,
            pos.y / self.screen_height,
            vel.x / 100,
            vel.y / 100,
            self.agent_body.angle,
            self.agent_body.angular_velocity,
            1.0 if self._is_grounded() else 0.0,
            min(dist_to_edge / 200, 1.0),  # Normalized to 200px ahead
            min(dist_to_next_platform / 200, 1.0),
        ], dtype=np.float32)
    
    def _is_grounded(self):
        """Check if agent is touching ground."""
        ground_y = self.screen_height - 50
        agent_bottom = self.agent_body.position.y + 15  # Half box height
        
        # Check if close to ground AND low vertical velocity AND on a platform
        is_low_enough = abs(agent_bottom - ground_y) < 8
        is_stable = abs(self.agent_body.velocity.y) < 100
        
        # Check if horizontally aligned with any platform
        on_platform = False
        agent_x = self.agent_body.position.x
        for body, shape, x1, x2 in self.ground_bodies:
            if x1 - 20 <= agent_x <= x2 + 20:  # Some horizontal tolerance
                on_platform = True
                break
        
        return is_low_enough and is_stable and on_platform
    
    def render(self):
        """Pygame rendering with visual enhancements."""
        if self.render_mode != "human":
            return
            
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()
            pygame.display.set_caption(f"Parkour RL - Difficulty {self.difficulty}")
        
        self.screen.fill((20, 20, 30))
        
        # Draw ground platforms
        for body, shape, x1, x2 in self.ground_bodies:
            p1 = shape.a + body.position
            p2 = shape.b + body.position
            pygame.draw.line(self.screen, (100, 200, 100), (p1.x, p1.y), (p2.x, p2.y), 10)
        
        # Draw gaps (visual indicators)
        ground_y = self.screen_height - 50
        for i in range(0, len(self.platform_edges) - 1, 2):
            gap_start = self.platform_edges[i]
            gap_end = self.platform_edges[i + 1] if i + 1 < len(self.platform_edges) else gap_start
            gap_width = gap_end - gap_start
            
            # Draw red danger zone
            pygame.draw.rect(
                self.screen,
                (80, 20, 20),
                (gap_start, ground_y, gap_width, 100),
                0
            )
            
            # Draw gap width text
            font = pygame.font.Font(None, 20)
            text = font.render(f"{gap_width:.0f}px", True, (200, 200, 200))
            self.screen.blit(text, (gap_start + gap_width/2 - 20, ground_y + 20))
        
        # Draw agent
        pos = self.agent_body.position
        angle = self.agent_body.angle
        vertices = [(v.rotated(angle) + pos) for v in self.agent_shape.get_vertices()]
        points = [(int(v.x), int(v.y)) for v in vertices]
        
        # Color based on grounded state
        color = (100, 255, 100) if self._is_grounded() else (255, 100, 100)
        pygame.draw.polygon(self.screen, color, points)
        
        # Draw velocity vector
        vel_scale = 2
        pygame.draw.line(
            self.screen,
            (255, 255, 100),
            (int(pos.x), int(pos.y)),
            (int(pos.x + self.agent_body.velocity.x * vel_scale), 
             int(pos.y + self.agent_body.velocity.y * vel_scale)),
            2
        )
        
        # Draw HUD
        font = pygame.font.Font(None, 30)
        hud_texts = [
            f"Step: {self.current_step}/{self.max_steps}",
            f"X: {pos.x:.0f}",
            f"Vel: ({self.agent_body.velocity.x:.0f}, {self.agent_body.velocity.y:.0f})",
            f"Grounded: {self._is_grounded()}",
            f"Difficulty: {self.difficulty}",
        ]
        
        for i, text in enumerate(hud_texts):
            surface = font.render(text, True, (200, 200, 200))
            self.screen.blit(surface, (10, 10 + i * 30))
        
        pygame.display.flip()
        self.clock.tick(60)
    
    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None