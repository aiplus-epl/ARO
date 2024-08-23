import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from custom_env import CustomRobotEnv

def run_ddpg(filtered_vertices, init_extrinsic, init_intrinsic):
    env = CustomRobotEnv(filtered_vertices, init_extrinsic, init_intrinsic)

    n_actions = env.action_space.shape[-1]
    ou_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # DDPG 모델 생성
    model = DDPG(
        policy="MultiInputPolicy",  # Dict 관찰 공간에 적합
        env=env,
        learning_rate=1e-4,
        buffer_size=500_000,
        learning_starts=10000,
        batch_size=256,
        tau=0.001,
        gamma=0.98,
        action_noise=ou_noise,
        train_freq=1,
        gradient_steps=100,
        verbose=1
    )

    # 모델 학습
    model.learn(total_timesteps=500)

    # 모델 저장
    model.save("ddpg_customenv")

    # 환경 초기화
    obs = env.reset()

    total_reward = 0
    episode_number = 0

    # 에피소드 실행
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        total_reward += reward
        if done:
            episode_number+=1
            obs = env.reset()