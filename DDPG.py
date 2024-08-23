import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from custom_env import CustomEnv

# Custom 환경 생성
env = CustomEnv()

# 행동 잡음 설정
n_actions = env.action_space.shape[-1]
ou_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# DDPG 모델 생성
model = DDPG("MlpPolicy", env, action_noise=ou_noise, verbose=1)

# 모델 학습
model.learn(total_timesteps=100000)

# 모델 저장
model.save("ddpg_customenv")

# 모델 불러오기
model = DDPG.load("ddpg_customenv")

# 환경 초기화
obs = env.reset()

# 에피소드 실행
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
