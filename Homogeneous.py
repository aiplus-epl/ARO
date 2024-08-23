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

    # 이동 행렬 생성
    translation_matrix = np.array([
        [1, 0, 0, px],
        [0, 1, 0, py],
        [0, 0, 1, pz],
        [0, 0, 0, 1]
    ])

    # 회전과 이동 적용
    transform_matrix = translation_matrix @ rotation_matrix @ matrix

    return transform_matrix
