import numpy as np


def homoCaculate(matrix, rx, ry, rz, px, py, pz):
    # Radians로 각도 변환
    rx = np.radians(rx)
    ry = np.radians(ry)
    rz = np.radians(rz)

    # 회전 행렬 생성
    rotation_x = np.array([
        [1, 0, 0, 0],
        [0, np.cos(rx), -np.sin(rx), 0],
        [0, np.sin(rx), np.cos(rx), 0],
        [0, 0, 0, 1]
    ])

    rotation_y = np.array([
        [np.cos(ry), 0, np.sin(ry), 0],
        [0, 1, 0, 0],
        [-np.sin(ry), 0, np.cos(ry), 0],
        [0, 0, 0, 1]
    ])

    rotation_z = np.array([
        [np.cos(rz), -np.sin(rz), 0, 0],
        [np.sin(rz), np.cos(rz), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # 전체 회전 행렬
    rotation_matrix = rotation_z @ rotation_y @ rotation_x

    # 이동 행렬 생성 (Z축 이동값을 0으로 설정)
    translation_matrix = np.array([
        [1, 0, 0, px],
        [0, 1, 0, py],
        [0, 0, 1, 0],  # Z축 이동을 없앰
        [0, 0, 0, 1]
    ])

    # 회전을 초기화한 상태로 이동 (회전 없이 이동)
    rotation_reset_matrix = np.identity(4)  # 회전 초기화
    transformed_matrix = rotation_reset_matrix @ matrix

    # 회전 초기화 후 이동 적용
    transformed_matrix = translation_matrix @ transformed_matrix

    # 새로운 회전 적용
    transformed_matrix = rotation_matrix @ transformed_matrix

    return transformed_matrix
