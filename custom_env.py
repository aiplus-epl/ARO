import gym
from gym import Env
from gym.spaces import Box
import numpy as np


# Custom 환경 정의
class CustomEnv(Env):
    def __init__(self):
        super().__init__()
        self.action_space = Box(low=np.array([-2]), high=np.array([2]), dtype=np.float32)
        self.observation_space = Box(low=np.array([0]), high=np.array([100]), dtype=np.float32)

        self.state = np.random.choice([-20, 0, 20, 40, 60])
        self.prev_state = self.state
        self.episode_length = 100

    def reset(self):
        self.state = np.random.choice([-20, 0, 20, 40, 60])
        self.episode_length = 100
        return self.get_obs()

    def get_obs(self):
        return np.array([self.state], dtype=int)

    def step(self, action):
        self.state += action[0]
        self.episode_length -= 1

        # Reward 책정
        if self.state >= 20 and self.state <= 25:
            reward = +100
        else:
            reward = -100

        prev_diff = min(abs(self.prev_state - 20), abs(self.prev_state - 25))
        curr_diff = min(abs(self.state - 20), abs(self.state - 25))

        if curr_diff <= prev_diff:
            if reward != 100:
                reward = reward + 50
            else:
                reward = 100
        if curr_diff > prev_diff:
            reward -= 50

        self.prev_state = self.state

        # Episode 끝났는지 확인
        done = self.episode_length <= 0
        info = {}

        return self.get_obs(), reward, done, info
