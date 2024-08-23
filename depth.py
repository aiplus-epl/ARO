import numpy as np
def getDepth(vertices, intrinsic, extrinsic):

    e00, e01, e02, e03 = extrinsic[0]
    e10, e11, e12, e13 = extrinsic[1]
    e20, e21, e22, e23 = extrinsic[2]
    e30, e31, e32, e33 = extrinsic[3]

    i00, i01, i02 = intrinsic[0]
    i10, i11, i12 = intrinsic[1]
    i20, i21, i22 = intrinsic[2]

    # 카메라 extrinsic 파라미터
    extrinsic = np.array([
        [e00, e01, e02, e03],
        [e10, e11, e12, e13],
        [e20, e21, e22, e23],
        [e30, e31, e32, e33]
    ])

    # 카메라 intrinsic 파라미터
    intrinsic = np.array([
        [i00, i01, i02],
        [i10, i11, i12],
        [i20, i21, i22]
    ])

    # 카메라 위치
    camera_position = np.array([e03, e13, e23])

    # 이미지 크기
    width = 1280
    height = 1024

    # 이미지 평면상의 좌표
    image_corners = np.array([
        [0, 0, 1],  # top-left
        [width, 0, 1],  # top-right
        [width, height, 1],  # bottom-right
        [0, height, 1]  # bottom-left
    ])

    # 카메라 좌표로 변환
    camera_corners = np.linalg.inv(intrinsic) @ image_corners.T
    camera_corners = camera_corners.T
    camera_corners = camera_corners[:, :3] / camera_corners[:, 2][:, np.newaxis]

    # 월드 좌표로 변환
    world_corners = (extrinsic @ np.hstack((camera_corners, np.ones((4, 1)))).T).T
    world_corners = world_corners[:, :3]

    # 카메라 위치
    camera_position = extrinsic[:3, 3]

    # 사다리꼴을 정의하는 4개의 평면 계산
    def plane_from_points(p1, p2, p3):
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)
        d = -np.dot(normal, p1)
        return np.append(normal, d)

    planes = [
        plane_from_points(camera_position, world_corners[0], world_corners[1]),
        plane_from_points(camera_position, world_corners[1], world_corners[2]),
        plane_from_points(camera_position, world_corners[2], world_corners[3]),
        plane_from_points(camera_position, world_corners[3], world_corners[0])
    ]

    # 각 정점이 사다리꼴 안에 있는지 확인
    def is_inside_frustum(point, planes):
        for plane in planes:
            if np.dot(plane[:3], point) + plane[3] > 0:
                return False
        return True

    # 사다리꼴 내부에 있는 정점들 필터링
    inside_vertices = [v for v in vertices if is_inside_frustum(v, planes)]
    camera_position = extrinsic[:3, 3]

    # 사다리꼴 내부에 있는 정점들을 카메라와의 거리로 정렬
    inside_vertices_sorted = sorted(inside_vertices, key=lambda v: np.linalg.norm(v - camera_position))
    delta_vertices = [v - camera_position for v in inside_vertices]
    return delta_vertices
