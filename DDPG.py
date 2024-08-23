import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from custom_env import CustomRobotEnv

def run_ddpg(filtered_vertices, init_extrinsic, init_intrinsic):
    # Custom 환경 생성
    env = CustomRobotEnv(filtered_vertices, init_extrinsic, init_intrinsic)

    # 행동 잡음 설정
    n_actions = env.action_space.shape[-1]
    ou_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # DDPG 모델 생성
    model = DDPG(
        policy="MultiInputPolicy",  # Dict 관찰 공간에 적합한 정책
        env=env,
        learning_rate=1e-4,  # 더 안정적인 학습을 위해 낮춘 학습률
        buffer_size=500_000,  # 더 큰 버퍼 크기
        learning_starts=10000,  # 더 많은 초기 경험 수집
        batch_size=256,  # 더 큰 배치 사이즈로 안정적인 학습
        tau=0.001,  # 타깃 네트워크의 더 느린 업데이트
        gamma=0.98,  # 약간 더 낮춘 할인율, 더 안정적인 학습
        action_noise=ou_noise,  # 탐색을 돕는 노이즈
        train_freq=1,  # 매 스텝마다 학습
        gradient_steps=100,  # 더 많은 그래디언트 스텝
        verbose=1  # 학습 과정의 출력을 상세하게
    )

    # 모델 학습
    model.learn(total_timesteps=500)  # 더 긴 학습 시간

    # 모델 저장
    model.save("ddpg_customenv")

    # 환경 초기화
    obs = env.reset()
    episode_number = 0  # 에피소드 번호 초기화

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