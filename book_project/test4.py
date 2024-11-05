import numpy as np
from itertools import permutations

def distance(point1, point2):
    """2点間の距離を計算する"""
    return np.linalg.norm(np.array(point1) - np.array(point2))

def get_camera_vectors(camera_position, A, B, C):
    """カメラからA、B、Cへのベクトルを計算する"""
    C_A = np.array(A) - np.array(camera_position)
    C_B = np.array(B) - np.array(camera_position)
    C_C = np.array(C) - np.array(camera_position)
    return C_A, C_B, C_C

def determine_rotation_direction(C_A, C_B, C_C):
    """カメラからのベクトルを用いて回転方向を判断する"""
    angle_AB = np.arctan2(C_B[1], C_B[0]) - np.arctan2(C_A[1], C_A[0])
    angle_BC = np.arctan2(C_C[1], C_C[0]) - np.arctan2(C_B[1], C_B[0])

    # 時計回りか反時計回りかを判断
    return "clockwise" if angle_BC < angle_AB else "counterclockwise"

def count_clockwise_turns(route, cameras, T):
    """コース上の各点に対する回転方向を数える"""
    counts = []
    
    for camera in cameras:
        camera_position = camera  # カメラの位置
        count = 0
        for i in range(len(route) - 2):
            A = route[i]
            B = route[i + 1]
            C = route[i + 2]
            C_A, C_B, C_C = get_camera_vectors(camera_position, A, B, C)
            direction = determine_rotation_direction(C_A, C_B, C_C)
            if direction == "clockwise":
                count += 1
        counts.append(count)
    
    return counts

def find_shortest_route(stars, cameras, T):
    """指定された条件を満たす最短のコースを探す"""
    n = len(stars)
    shortest_length = float('inf')
    best_route = None

    for route in permutations(stars):
        total_distance = sum(distance(route[i], route[i + 1]) for i in range(n - 1))
        counts = count_clockwise_turns(route, cameras, T)
        print(route,counts)

        if all(count == T[j] for j, count in enumerate(counts)):
            if total_distance < shortest_length:
                shortest_length = total_distance
                best_route = route

    return shortest_length, best_route

# 入力の例
n = 4
m = 2
stars = [
    (-3, 5, -3),
    (-2, -3, 5),
    (1, -1, 0),
    (-5, -2, -4)
]
cameras = [
    (0, 0, -2),  # z座標だけを考慮
    (0, 0, -3)
]
T = [2, 2]

# 最短コースを探す
shortest_length, best_route = find_shortest_route(stars, cameras, T)
print("最短コースの長さ:", shortest_length)
print("最短コースのルート:", best_route)