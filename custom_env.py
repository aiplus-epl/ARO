import gym
from gym import spaces
import numpy as np
from blocked import isBlocked  # 충돌 체크 함수
from depth import getDepth  # Depth 계산 함수
from homogeneous import homoCaculate  # 호모그래피 계산 함수



class CustomRobotEnv(gym.Env):
    def __init__(self, filtered_vertices, init_extrinsic, init_intrinsic, depth_features_max_points=1000):
        super(CustomRobotEnv, self).__init__()

        # 환경에서 사용할 필터링된 정점들
        self.filtered_vertices = filtered_vertices

        # 초기 extrinsic 및 intrinsic 파라미터
        self.init_extrinsic = np.array(init_extrinsic).reshape(4, 4)
        self.init_intrinsic = np.array(init_intrinsic).reshape(3, 3)

        # Depth features의 최대 점 수 설정
        self.depth_features_max_points = depth_features_max_points

        self.episode = 0

        # 행동 공간 정의 (x, y, z 축 회전 및 이동)
        self.action_space = spaces.Box(
            low=np.array([-90, -90, -90, -0.05, -0.05, -0.05]),
            high=np.array([90, 90, 90, 0.05, 0.05, 0.05]),
            dtype=np.float32
        )

        # 상태 공간 정의 (이동 거리, 방문 노드, 이동 횟수, Depth Features)
        self.observation_space = spaces.Dict({
            "distance": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "visited_nodes": spaces.Box(low=-np.inf, high=np.inf, shape=(100, 4, 4), dtype=np.float32),
            # 최대 100개의 4x4 행렬
            "step_count": spaces.Discrete(100),
            "depth_features": spaces.Box(low=-np.inf, high=np.inf, shape=(self.depth_features_max_points, 3),
                                         dtype=np.float32)  # [x, y, z] 형식의 점들로 이루어진 배열
        })

        # 초기 상태 설정
        self.reset()

    def reset(self):
        # 환경 초기화 및 초기 상태 반환
        self.current_position = self.init_extrinsic.copy()  # 초기 extrinsic 행렬로 위치 초기화
        self.visited_nodes = np.zeros((0, 4, 4))  # 초기에는 방문한 노드가 없음
        self.step_count = 0
        self.done = False
        self.collision_count = 0  # 충돌 횟수 초기화
        self.total_reward = 0  # 에피소드 동안의 누적 보상

        # 카메라 intrinsic 및 extrinsic 파라미터 설정
        self.extrinsic = self.init_extrinsic.copy()
        self.intrinsic = self.init_intrinsic.copy()

        # 초기 Depth Features 계산
        depth_features = self._get_padded_depth_features(
            getDepth(self.filtered_vertices, self.intrinsic, self.extrinsic))

        # 상태 초기화 (예: 현재까지 이동한 거리, 방문한 노드, 이동 횟수, Depth Features)
        initial_state = {
            "distance": np.array([0.0]),
            "visited_nodes": np.zeros((100, 4, 4)),  # [4x4 행렬] 형식으로 최대 100개 노드
            "step_count": self.step_count,
            "depth_features": depth_features  # [x, y, z] 형식의 점들로 된 배열
        }

        return initial_state

    def step(self, action):
        if self.done:
            print("Episode should have ended, done is True.")
            return self.reset(), 0, True, {}

        # 이전 위치 저장
        previous_position = self.current_position.copy()

        # action에 따른 로봇의 회전 및 이동 적용
        rx, ry, rz, px, py, pz = action
        new_position = homoCaculate(self.current_position, rx, ry, rz, px, py, pz)

        # 새로운 위치의 (x, y) 좌표 추출
        current_xy = self.current_position[:2, 3]
        new_xy = new_position[:2, 3]

        # 충돌 감지
        closest_point = isBlocked(self.filtered_vertices, *current_xy, *new_xy)
        if closest_point is not None:
            # 충돌 발생 시, 위치를 이전 위치로 되돌림
            self.current_position = previous_position
            self.collision_count += 1  # 충돌 횟수 증가
            reward = -1 # 충돌 시 마이너스 보상

            # 충돌 횟수가 10번 이상이면 에피소드 종료
            if self.collision_count >= 5:
                self.done = True
        else:
            # 새로운 위치가 이전에 방문한 노드와 가까운지 확인
            if self._is_close_to_visited_node(new_position):
                reward = -0.02  # 중복 방문 시 마이너스 보상
            else:
                reward = np.linalg.norm(new_xy - current_xy) * 2 # 이동 거리에 따른 보상
                self.visited_nodes = np.vstack([self.visited_nodes, [new_position]])  # 새로운 노드 추가
                # 새로운 위치를 현재 위치로 업데이트
                self.current_position = new_position

        self.total_reward += reward  # 보상 누적
        self.step_count += 1

        # Depth 정보 갱신
        depth_features = self._get_padded_depth_features(
            getDepth(self.filtered_vertices, self.intrinsic, self.extrinsic))

        # 종료 조건 확인
        self.done = self.done or self.step_count >= 100

        # 새 상태 반환
        state = {
            "distance": np.array([np.linalg.norm(self.current_position[:3, 3])]),  # 현재 위치의 3D 좌표 거리
            "visited_nodes": self._get_padded_visited_nodes(),
            "step_count": self.step_count,
            "depth_features": depth_features  # 업데이트된 Depth Features
        }

        # 스텝 정보 출력
        print(f"Step {self.step_count} - Position: {self.current_position[:3, 3]}, Reward: {reward}, Total Reward: {self.total_reward}, Done: {self.done}")



        with open("reward.txt", "a") as file:
            file.write(f"{self.episode}\t{float(self.current_position[0, 3])}\t{float(self.current_position[1, 3])}\t{float(self.current_position[2, 3])}\t{reward}\t{self.total_reward}\t{self.done}\n")

        if self.done:
            self.episode +=1

        return state, reward, self.done, {}

    def _get_padded_depth_features(self, depth_features):
        # Depth Features를 최대 점 수로 패딩하여 반환
        padding_value = 100  # 패딩을 위해 사용할 큰 값
        if len(depth_features) == 0:
            return np.full((self.depth_features_max_points, 3), padding_value)

        if len(depth_features) < self.depth_features_max_points:
            padding = np.full((self.depth_features_max_points - len(depth_features), 3), padding_value)
            return np.vstack([depth_features, padding])
        else:
            return depth_features[:self.depth_features_max_points]

    def _is_close_to_visited_node(self, position, threshold=0.1):
        # 이전에 방문한 노드와 현재 위치가 가까운지 확인
        if self.visited_nodes.shape[0] == 0:
            return False
        distances = np.linalg.norm(self.visited_nodes[:, :3, 3] - position[:3, 3], axis=1)
        return np.any(distances < threshold)

    def _get_padded_visited_nodes(self):
        # 방문한 노드 배열을 반환, 최대 크기로 패딩
        if len(self.visited_nodes) < 100:
            padding = np.zeros((100 - len(self.visited_nodes), 4, 4))
            return np.vstack([self.visited_nodes, padding])
        else:
            return self.visited_nodes[:100]

    def render(self, mode="human"):
        # 시각화 기능 (원할 경우 구현)
        pass

    def close(self):
        # 종료 시 처리할 작업
        pass
