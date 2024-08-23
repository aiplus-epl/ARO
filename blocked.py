import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

def isBlocked(filtered_vertices, x1, y1, x2, y2):
    point1 = np.array([x1, y1])  # 현재점
    point2 = np.array([x2, y2])  # 이후점

    # 두 점 사이 넣을 점 개수
    num_points = 100000
    line_x = np.linspace(point1[0], point2[0], num_points)
    line_y = np.linspace(point1[1], point2[1], num_points)
    line_points = np.vstack((line_x, line_y)).T

    # k-d 트리를 사용해 필터링된 정점과 선 사이의 최소 거리를 계산
    tree = cKDTree(filtered_vertices[:, :2])
    distances, indices = tree.query(line_points)

    threshold = 0.001  # 임계값

    # 최소 거리 계산
    close_indices = np.where(distances <= threshold)[0]

    # 맞닿는 부분이 있다면 yes, 없다면 no
    if len(close_indices) > 0:
        # 맞닿는 좌표를 표시
        intersecting_points = line_points[close_indices]
        # x1, y1과 가장 가까운 맞닿는 점 찾기
        distances_to_point1 = np.linalg.norm(intersecting_points - point1, axis=1)
        closest_point_index = np.argmin(distances_to_point1)
        closest_point = intersecting_points[closest_point_index]
        return closest_point
    else:
        return None
    #
    #     # 전체 그래프를 시각화
    #     plt.figure(figsize=(10, 10))
    #     plt.scatter(filtered_vertices[:, 0], filtered_vertices[:, 1], s=0.5, label='Filtered Points')
    #     plt.plot(line_x, line_y, color='red', linewidth=1, label='Given Line')
    #     plt.scatter(intersecting_points[:, 0], intersecting_points[:, 1], color='yellow', s=30, label='Intersection Points')
    #     plt.scatter(closest_point[0], closest_point[1], color='red', s=30, label='Closest Intersection to Point1')
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     plt.title('Line Intersection with Grid Map')
    #     plt.axis('equal')
    #     plt.legend()
    #     plt.show()
    #     print(f"Closest intersecting point to Point1: {closest_point}")
    # else:
    #     print("no")
    #
    #     # 맞닿지 않은 경우에도 선과 필터링된 점들을 시각화
    #     plt.figure(figsize=(10, 10))
    #     plt.scatter(filtered_vertices[:, 0], filtered_vertices[:, 1], s=0.5, label='Filtered Points')
    #     plt.plot(line_x, line_y, color='red', linewidth=1, label='Given Line')
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     plt.title('Line and Grid Map')
    #     plt.axis('equal')
    #     plt.legend()
    #     plt.show()
