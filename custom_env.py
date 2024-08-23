import gym
from gym import spaces
import numpy as np
from blocked import isBlocked
from depth import getDepth
from homogeneous import homoCaculate


class CustomRobotEnv(gym.Env):
    def __init__(self, filtered_vertices):
        super(CustomRobotEnv, self).__init__()

        # 환경에서 사용할 필터링된 정점들
        self.filtered_vertices = filtered_vertices

        # 행동 공간 정의 (x, y, z 축 회전 및 이동)
        self.action_space = spaces.Box(
            low=np.array([-180, -180, -180, -0.3, -0.3, -0.3]),
            high=np.array([180, 180, 180, 0.3, 0.3, 0.3]),
            dtype=np.float32
        )

        # 상태 공간 정의 (이동 거리, 방문 노드, 이동 횟수, Depth Features)
        self.observation_space = spaces.Dict({
            "distance": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "visited_nodes": spaces.Box(low=-np.inf, high=np.inf, shape=(100, 4, 4), dtype=np.float32),
            # 최대 100개의 4x4 행렬
            "step_count": spaces.Discrete(100),
            "depth_features": spaces.Box(low=0, high=1, shape=(100,), dtype=np.float32)
        })

        # 초기 상태 설정
        self.reset()

    def reset(self):
        # 환경 초기화 및 초기 상태 반환
        self.current_position = np.eye(4)  # 초기 위치는 단위 행렬로 설정
        self.visited_nodes = np.zeros((0, 4, 4))  # 초기에는 방문한 노드가 없음
        self.step_count = 0
        self.done = False

        # 카메라 파라미터 설정
        self.intrinsic = np.array([
            [640, 0, 320],
            [0, 640, 240],
            [0, 0, 1]
        ])
        self.extrinsic = np.eye(4)  # 초기 값으로 단위 행렬 사용

        # 상태 초기화 (예: 현재까지 이동한 거리, 방문한 노드, 이동 횟수, Depth Features)
        initial_state = {
            "distance": np.array([0.0]),
            "visited_nodes": np.zeros((100, 4, 4)),  # [4x4 행렬] 형식으로 최대 100개 노드
            "step_count": self.step_count,
            "depth_features": np.random.random(100)
        }

        return initial_state

    def step(self, action):
        if self.done:
            return self.reset(), 0, True, {}

        # action에 따른 로봇의 회전 및 이동 적용
        rx, ry, rz, px, py, pz = action
        new_position = homoCaculate(self.current_position, rx, ry, rz, px, py, pz)

        # 새로운 위치의 (x, y) 좌표 추출
        current_xy = self.current_position[:2, 3]
        new_xy = new_position[:2, 3]

        # 충돌 감지
        closest_point = isBlocked(self.filtered_vertices, *current_xy, *new_xy)
        if closest_point is not None:
            self.done = True
            reward = -100  # 충돌 시 큰 마이너스 보상
        else:
            # 새로운 위치가 이전에 방문한 노드와 가까운지 확인
            if self._is_close_to_visited_node(new_position):
                reward = -10  # 중복 방문 시 마이너스 보상
            else:
                reward = np.linalg.norm(new_xy - current_xy)  # 이동 거리에 따른 보상
                self.visited_nodes = np.vstack([self.visited_nodes, [new_position]])  # 새로운 노드 추가

        # 상태 업데이트
        self.current_position = new_position
        self.step_count += 1

        # Depth 정보 갱신
        depth_features = getDepth(self.filtered_vertices, self.intrinsic, self.extrinsic)

        # 종료 조건 확인
        self.done = self.done or self.step_count >= 100

        # 새 상태 반환
        state = {
            "distance": np.array([np.linalg.norm(self.current_position[:3, 3])]),  # 현재 위치의 3D 좌표 거리
            "visited_nodes": self._get_padded_visited_nodes(),
            "step_count": self.step_count,
            "depth_features": depth_features  # 업데이트된 Depth Features
        }

        return state, reward, self.done, {}

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
