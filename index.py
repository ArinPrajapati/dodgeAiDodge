import pygame as pg
import sys, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

#####################
# Environment Class #
#####################

class DodgeEnv:
    def __init__(self, render_mode=False):
        self.WIDTH = 500
        self.HEIGHT = 500
        self.BACKGROUND = (234, 212, 252)
        self.FALLING_COLOR = (0, 233, 255)
        self.PLAYER_COLOR = (255, 24, 24)
        self.SIZE = 30
        self.fps = 60
        self.danger_zone = 100  # pixels above the player considered "danger"
        self.render_mode = render_mode

        if self.render_mode:
            pg.init()
            self.window = pg.display.set_mode((self.WIDTH, self.HEIGHT))
            pg.display.set_caption("Dodge AI Dodge")
            self.clock = pg.time.Clock()

        self.reset()

    def reset(self):
        self.player_x = random.randrange(1, self.WIDTH - self.SIZE)
        self.player_y = self.HEIGHT - self.SIZE - 10
        self.falling_objects = []  # Each object: {"rect": pg.Rect, "speed": int, "rewarded": bool}
        self.time_elapsed = 0.0  # seconds elapsed in the episode
        return self.get_state()

    def get_state(self):
        """
        The state is represented as a vector:
          [player_x/WIDTH, player_y/HEIGHT, nearest_obj_x/WIDTH, nearest_obj_y/HEIGHT, nearest_obj_speed/5]
        If no falling objects exist, defaults to zeros.
        """
        nearest = None
        min_dist = float('inf')
        for obj in self.falling_objects:
            # Only consider objects that are above the player (i.e. falling toward the player)
            dist = self.player_y - obj["rect"].y  
            if dist >= 0 and dist < min_dist:
                min_dist = dist
                nearest = obj

        if nearest is None:
            nearest_x = 0.0
            nearest_y = 0.0
            nearest_speed = 0.0
        else:
            nearest_x = nearest["rect"].x / self.WIDTH
            nearest_y = nearest["rect"].y / self.HEIGHT
            nearest_speed = nearest["speed"] / 5.0  # assuming maximum speed is 5

        state = np.array([self.player_x / self.WIDTH,
                          self.player_y / self.HEIGHT,
                          nearest_x,
                          nearest_y,
                          nearest_speed], dtype=np.float32)
        return state

    def step(self, action):
        """
        Actions: 0 = stay, 1 = move left, 2 = move right.
        Returns: next_state, reward, done, info.
        """
        reward = 0.0
        # Add time reward (per frame)
        reward += 0.01 / self.fps
        # Penalty for staying in place (discourage inactivity)
        if action == 0:
            reward -= 0.08 / self.fps

        # Update player's position
        if action == 1:
            self.player_x = max(0, self.player_x - 1)
        elif action == 2:
            self.player_x = min(self.WIDTH - self.SIZE, self.player_x + 1)

        # Update falling objects
        for obj in self.falling_objects:
            obj["rect"].y += obj["speed"]
            # When a falling object enters the danger zone (and hasn't been rewarded yet), give bonus.
            if (not obj["rewarded"]) and (obj["rect"].y >= self.player_y - self.danger_zone):
                reward += 4.0
                obj["rewarded"] = True

        # Remove objects that have fallen off-screen.
        self.falling_objects = [obj for obj in self.falling_objects if obj["rect"].y < self.HEIGHT]

        # Occasionally spawn a new falling object (adjust the probability as needed)
        if random.random() < 0.03:
            x = random.randint(0, self.WIDTH - 20)
            y = 0
            speed = random.randint(2, 5)
            self.falling_objects.append({"rect": pg.Rect(x, y, 20, 20),
                                         "speed": speed,
                                         "rewarded": False})

        # Check for collisions. (A collision ends the episode.)
        player_rect = pg.Rect(self.player_x, self.player_y, self.SIZE, self.SIZE)
        done = False
        for obj in self.falling_objects:
            if player_rect.colliderect(obj["rect"]):
                done = True
                reward -= 100 # heavy penalty for collision
                break

        self.time_elapsed += 1 / self.fps
        next_state = self.get_state()
        return next_state, reward, done, {}

    def render(self):
        if not self.render_mode:
            return
        # Draw background
        self.window.fill(self.BACKGROUND)
        # Draw falling objects
        for obj in self.falling_objects:
            pg.draw.rect(self.window, self.FALLING_COLOR, obj["rect"])
        # Draw player
        player_rect = pg.Rect(self.player_x, self.player_y, self.SIZE, self.SIZE)
        pg.draw.rect(self.window, self.PLAYER_COLOR, player_rect)
        pg.display.update()
        self.clock.tick(self.fps)

#############################
# DQN Agent Implementation  #
#############################

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        
    def push(self, transition):
        self.memory.append(transition)
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

#############################
# Training Loop and Agent   #
#############################

def train_dqn():
    # Create environment. Set render_mode=True to see the game.
    env = DodgeEnv(render_mode=True)
    state_dim = 5    # [player_x, player_y, nearest_obj_x, nearest_obj_y, nearest_obj_speed]
    action_dim = 3   # 0 = stay, 1 = left, 2 = right

    # Create the policy and target networks.
    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    memory = ReplayMemory(10000)
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.995

    num_episodes = 500
    update_target_every = 10

    for i_episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        total_reward = 0.0
        done = False
        step_count = 0

        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randrange(action_dim)
            else:
                with torch.no_grad():
                    q_values = policy_net(state)
                    action = q_values.argmax().item()

            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            memory.push((state, action, reward, next_state_tensor, done))
            state = next_state_tensor

            # Sample random mini-batch and perform a training step if we have enough samples
            if len(memory) >= batch_size:
                transitions = memory.sample(batch_size)
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
                batch_state = torch.cat(batch_state)
                batch_action = torch.tensor(batch_action).unsqueeze(1)
                batch_reward = torch.tensor(batch_reward, dtype=torch.float32).unsqueeze(1)
                batch_next_state = torch.cat(batch_next_state)
                batch_done = torch.tensor(batch_done, dtype=torch.float32).unsqueeze(1)

                current_q = policy_net(batch_state).gather(1, batch_action)
                with torch.no_grad():
                    max_next_q = target_net(batch_next_state).max(1)[0].unsqueeze(1)
                    expected_q = batch_reward + gamma * max_next_q * (1 - batch_done)
                loss = nn.MSELoss()(current_q, expected_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if env.render_mode:
                env.render()
            step_count += 1

        print(f"Episode {i_episode} Total Reward: {total_reward:.2f}  Steps: {step_count}")
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if i_episode % update_target_every == 0:
            target_net.load_state_dict(policy_net.state_dict())

    if env.render_mode:
        pg.quit()
    sys.exit()

if __name__ == "__main__":
    train_dqn()

