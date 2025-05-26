import numpy as np
import sys
import os
from .server_tools import *
from .math_tools import *
#bezier cutting

#perpendicular_distance:計算點到線段的垂直距離
def perpendicular_distance(point, line_start, line_end):
    """計算點到線段的垂直距離"""
    if np.array_equal(line_start, line_end):
        return np.linalg.norm(point - line_start)
    
    line_vec = line_end - line_start
    point_vec = point - line_start
    
    line_len = np.dot(line_vec, line_vec)
    t = max(0, min(1, np.dot(point_vec, line_vec) / line_len))
    projection = line_start + t * line_vec
    return np.linalg.norm(point - projection)
#rdp:道格拉斯-普克線簡化演算法簡化點位
def rdp(points, epsilon):
    """Douglas-Peucker 道格拉斯-普克線簡化演算法"""
    points = np.array(points)
    
    if len(points) < 3:
        return points.tolist()
    
    start, end = points[0], points[-1]
    max_dist = 0
    index = 0
    
    for i in range(1, len(points) - 1):
        dist = perpendicular_distance(points[i], start, end)
        if dist > max_dist:
            max_dist = dist
            index = i
    
    if max_dist > epsilon:
        left_simplified = rdp(points[:index+1], epsilon)
        right_simplified = rdp(points[index:], epsilon)
        
        return left_simplified[:-1] + right_simplified
    else:
        return [start.tolist(), end.tolist()]
#path_simplify_and_extract:自創路徑演算法提取特徵點
def svcfp(paths, min_radius=10, max_radius=50, curvature_threshold=27, rdp_epsilon=2, ifserver=1):
    import numpy as np

    paths = np.array(paths)
    simplified_points = rdp(paths, rdp_epsilon)
    custom_print(ifserver, f"rdp 簡化後的點數: {len(simplified_points)}")

    original_indices = []
    i = 0
    check_paths_idx = 0
    while check_paths_idx < len(paths) and i < len(simplified_points):
        if np.array_equal(paths[check_paths_idx], simplified_points[i]):
            original_indices.append(check_paths_idx)
            i += 1
        check_paths_idx += 1

    if i < len(simplified_points):
        custom_print(ifserver, "警告: simplified_points 中剩餘的元素無法在 paths 中按順序找到")

    stdlist, max_values, angle = [], [], []
    all_feature_values = []

    def calculate_angle_change(p1, p2, p3):
        if np.array_equal(p1, p2) or np.array_equal(p2, p3):
            return 0
        v1 = p1 - p2
        v2 = p3 - p2
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        angle = np.dot(v1_norm, v2_norm)
        return angle

    def cross_sign(p1, p2, p3):
        v1 = p2 - p1
        v2 = p3 - p2
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        return np.sign(cross)

    prev_cross_sign = None

    for i in range(len(original_indices)):
        original_idx = original_indices[i]
        angle_change = 0
        cross_sign_change = False

        if i > 1 and i < len(original_indices) - 1:
            A = paths[original_indices[i - 2]]
            B = paths[original_indices[i - 1]]
            C = paths[original_indices[i]]
            D = paths[original_indices[i + 1]]

            sign1 = cross_sign(A, B, C)
            sign2 = cross_sign(B, C, D)

            if sign1 != 0 and sign2 != 0 and sign1 != sign2:
                cross_sign_change = True

        if i > 0 and i < len(original_indices) - 1:
            prev_idx = original_indices[i - 1]
            next_idx = original_indices[i + 1]
            angle_change = calculate_angle_change(paths[prev_idx], paths[original_idx], paths[next_idx])
            angle.append(angle_change)
        else:
            angle.append(0)

        std_values = []
        max_distances = []

        for j in range(min_radius, max_radius):
            temp = []
            left_count, right_count = 0, 0

            k = 0
            while original_idx + k < len(paths) and distance(paths[original_idx], paths[original_idx + k]) < j:
                temp.append(paths[original_idx + k])
                right_count += 1
                k += 1

            k = 1
            while original_idx - k >= 0 and distance(paths[original_idx], paths[original_idx - k]) < j:
                temp.append(paths[original_idx - k])
                left_count += 1
                k += 1

            diff = abs(left_count - right_count)
            if diff > 0:
                if left_count > right_count:
                    k = right_count
                    while original_idx + k < len(paths) and distance(paths[original_idx], paths[original_idx + k]) < j and diff > 0:
                        temp.append(paths[original_idx + k])
                        k += 1
                        diff -= 1
                else:
                    k = left_count
                    while original_idx - k >= 0 and distance(paths[original_idx], paths[original_idx - k]) < j and diff > 0:
                        temp.append(paths[original_idx - k])
                        k += 1
                        diff -= 1

            if len(temp) < 3:
                continue

            temp = np.array(temp)
            std_value = np.std(temp, axis=0)
            std_values.append(np.mean(std_value))
            avg_coords = np.mean(temp, axis=0)
            max_dist = np.max([distance(avg_coords, p) for p in temp])
            max_distances.append(max_dist)

        if std_values and max_distances:
            mean_std = np.mean(std_values)
            mean_max_dist = np.mean(max_distances)
            combined_value = 0.3 * mean_std + 0.7 * mean_max_dist
            combined_value *= (2 + angle_change)

            if cross_sign_change:
                combined_value *= 1.1  # 增加權重

            stdlist.append(mean_std)
            max_values.append(combined_value)

            print(f"點 {i} (原始索引 {original_idx}): 標準差={mean_std:.2f}, 最大距離={mean_max_dist:.2f}, 角度變化={angle_change:.2f}, 加權值={combined_value:.2f}")

        else:
            stdlist.append(0)
            max_values.append(0)

        all_feature_values.append({
            'index': i,
            'original_index': original_idx,
            'position': simplified_points[i],
            'value': max_values[-1],
            'angle': angle[-1]
        })

    candidate_breakpoints = [i for i, val in enumerate(max_values) if val > curvature_threshold]

    if len(simplified_points) > 0:
        if 0 not in candidate_breakpoints:
            candidate_breakpoints.insert(0, 0)
        if len(simplified_points) - 1 not in candidate_breakpoints:
            candidate_breakpoints.append(len(simplified_points) - 1)

    final_idx = []
    i = j = 0
    while j < len(paths) and i < len(candidate_breakpoints):
        cb_idx = candidate_breakpoints[i]
        if np.array_equal(paths[j], simplified_points[cb_idx]):
            final_idx.append(j)
            i += 1
        j += 1

    key_points = [paths[idx] for idx in final_idx]

    custom_print(ifserver, f"找到 {len(key_points)} 個關鍵點")
    custom_print(ifserver, f"關鍵點原始 index: {final_idx}")
    custom_print(ifserver, f"關鍵點座標: {key_points}")

    return key_points, final_idx

def filter_key_points(key_points, indices, max_values, min_distance=10, min_value_ratio=0.5):
    """過濾過近的關鍵點"""
    if len(key_points) <= 2:  # 保留起點和終點
        return key_points, indices
    
    # 起點和終點一定保留
    filtered_points = [key_points[0]]
    filtered_indices = [indices[0]]
    
    # 處理中間點，只過濾過近的點
    for i in range(1, len(key_points) - 1):
        # 檢查與前一個保留點的距離
        dist_to_prev = distance(key_points[i], filtered_points[-1])
        
        # 如果距離太近，則跳過
        if dist_to_prev < min_distance:
            continue
        
        filtered_points.append(key_points[i])
        filtered_indices.append(indices[i])
    
    # 添加終點（如果不是起點）
    if len(key_points) > 1 and not np.array_equal(key_points[0], key_points[-1]):
        filtered_points.append(key_points[-1])
        filtered_indices.append(indices[-1])
    
    return filtered_points, filtered_indices

def calculate_cross_product_direction_change(p1, p2, p3):
    """
    判斷三個點形成的兩個向量之間的外積方向是否改變
    比較 v1 × v2 和 v2 × v3 的符號
    
    參數:
    p1, p2, p3: 三個連續點
    
    返回:
    1: 如果外積方向改變（向左轉或向右轉）
    0: 如果外積為零（點共線）或方向不變
    """
    # 計算向量
    v1 = p2 - p1
    v2 = p3 - p2
    
    # 計算外積（在2D平面上，外積是一個標量）
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]
    
    # 如果外積為零，則返回0（表示共線）
    if abs(cross_product) < 1e-10:
        return 0
    
    # 判斷外積的符號
    # 如果為正，表示向量v1到v2是逆時針旋轉
    # 如果為負，表示向量v1到v2是順時針旋轉
    # 這裡我們只返回1，因為有轉向
    return 1

def calculate_angle_change(p1, p2, p3):
    """計算三點間的夾角（改進版）"""
    if np.array_equal(p1, p2) or np.array_equal(p2, p3):
        return 0
    
    v1 = p2 - p1
    v2 = p3 - p2
    
    # 計算單位向量
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    if v1_norm == 0 or v2_norm == 0:
        return 0
    
    v1_unit = v1 / v1_norm
    v2_unit = v2 / v2_norm
    
    # 計算夾角的餘弦值
    dot_product = np.dot(v1_unit, v2_unit)
    # 確保在有效範圍 [-1, 1]
    dot_product = max(-1.0, min(1.0, dot_product))
    
    # 計算角度（弧度）
    angle_rad = np.arccos(dot_product)
    # 轉換為角度
    angle_deg = np.degrees(angle_rad)
    
    # 返回角度，越小代表轉角越大（余弦值越小）
    return angle_deg

def svcfp_queue(paths, simplified_points, min_radius=10, max_radius=50, curvature_threshold=27, debug=False,ifserver=1):
    """
    改進版路徑簡化和關鍵點提取演算法，解決偽關鍵點問題，增強尖銳角點檢測
    
    步驟：
    1. 使用 rdp 演算法簡化整個路徑
    2. 將簡化後的點映射回原始路徑的索引
    3. 計算每個點的曲率特徵值和角度變化
    4. 使用多種過濾方法去除偽關鍵點
    5. 增加直接針對尖銳角點的檢測
    6. 確保起點和終點被包含
    7. 最後進行密度過濾，消除過近的關鍵點
    
    參數:
        paths: 原始路徑點列表
        simplified_points: 已簡化的路徑點列表
        min_radius: 搜索範圍的最小格數
        max_radius: 搜索範圍的最大格數
        curvature_threshold: 判斷關鍵點的閾值
        debug: 是否打印詳細除錯信息
    
    返回:
        關鍵點座標列表 [[x1, y1], [x2, y2], ...]
        關鍵點在原始路徑中的索引列表
    """
    paths = np.array(paths)
    simplified_points = np.array(simplified_points)

    
    # 自動依據長度調整搜尋格數
    length = len(paths)
    if length <= 200:
        min_radius = max(3, int(min_radius * (length/200)))
        max_radius = max(10, int(max_radius * (length/200)))
    
    # 將簡化後的點映射回原始路徑的索引
    # 確保original_indices是整數列表，而不是點的列表
    original_indices = find_simplified_indices(paths, simplified_points)
    """
    print(simplified_points)
    for i in range(len(paths)):
        print(paths[original_indices[i]])
    """
    
    # 計算各點的標準差、極值數據和角度變化
    stdlist = []
    max_values = []
    angle = []
    all_feature_values = []  # 存儲所有點的特徵值，用於可視化
    
    # 為了計算角度變化，需要在簡化路徑上有三個連續點
    for i in range(len(original_indices)):
        # 獲取簡化後路徑中的當前點索引
        original_idx = original_indices[i]
        
        # 計算角度變化（如果有前後點）
        angle_change = 0
        if i > 0 and i < len(original_indices) - 1:
            prev_idx = original_indices[i-1]
            next_idx = original_indices[i+1]
            angle_change = calculate_angle_change(paths[prev_idx], paths[original_idx], paths[next_idx])
            angle.append(angle_change)
        else:
            angle.append(0)
        
        std_values = []
        max_distances = []
        
        # 修改為使用固定格數而非距離半徑
        for step_size in range(min_radius, max_radius):
            temp = []
            # 加入當前點
            temp.append(paths[original_idx])
            
            # 向右走step_size格
            for k in range(1, step_size + 1):
                right_idx = make_circular_index(original_idx + k, length)
                temp.append(paths[right_idx])
            
            # 向左走step_size格
            for k in range(1, step_size + 1):
                left_idx = make_circular_index(original_idx - k, length)
                temp.append(paths[left_idx])

            if len(temp) < 3:
                continue

            temp = np.array(temp)
            std_value = np.std(temp, axis=0)
            std_values.append(np.mean(std_value))

            avg_coords = np.mean(temp, axis=0)
            max_dist = np.max([distance(avg_coords, p) for p in temp])
            max_distances.append(max_dist)

        if len(std_values) > 0 and len(max_distances) > 0:
            # 取標準差和最大距離的均值
            mean_std = np.mean(std_values)
            mean_max_dist = np.mean(max_distances)
            
            # 使用原始的權重比例
            angle_weight = 0.4
            std_weight = 0.3
            dist_weight = 0.7
            
            # 使用角度變化值直接作為加權參數（越大越尖銳）
            angle_factor = 1.0 + (angle_change / 180.0)  # 將角度映射到[1, 2]範圍
            
            combined_value = (
                std_weight * mean_std + 
                dist_weight * mean_max_dist + 
                angle_weight * angle_change
            ) * angle_factor*10/14
            
            stdlist.append(mean_std)
            max_values.append(combined_value)
            
            if debug:
                print(f"點 {i} (原始索引 {original_idx}): 標準差={mean_std:.2f}, 最大距離={mean_max_dist:.2f}, " +
                      f"角度變化={angle_change:.2f}, 加權值={combined_value:.2f}")
        else:
            # 處理找不到足夠點的情況
            stdlist.append(0)
            max_values.append(0)
            if debug:
                print(f"點 {i} (原始索引 {original_idx}): 沒有足夠的點進行計算")
        
        # 存儲所有點的特徵值，包括索引和位置信息
        all_feature_values.append({
            'index': i,
            'original_index': original_idx,
            'position': paths[original_idx].tolist() if isinstance(paths[original_idx], np.ndarray) else paths[original_idx],
            'value': max_values[-1] if max_values else 0,
            'angle': angle[-1] if angle else 0,
        })
    
    # 尋找加權值超過閾值的點作為候選關鍵點
    candidate_breakpoints = []
    for i in range(len(max_values)):
        if max_values[i] > curvature_threshold:
            candidate_breakpoints.append(i)
            if debug:
                print(f"候選點 {i} (原始索引 {original_indices[i]}): 加權值={max_values[i]:.2f} > 閾值={curvature_threshold}")

    # 確保包含起點和終點
    if len(simplified_points) > 0:
        if 0 not in candidate_breakpoints:
            candidate_breakpoints.insert(0, 0)
            if debug:
                print("添加起點 0 作為關鍵點")
        if len(simplified_points) - 1 not in candidate_breakpoints:
            candidate_breakpoints.append(len(simplified_points) - 1)
            if debug:
                print(f"添加終點 {len(simplified_points) - 1} 作為關鍵點")
    
    # 將候選關鍵點從簡化路徑轉換回原始路徑
    final_idx = [original_indices[bp] for bp in candidate_breakpoints]
    
    # 排序索引以保持有序
    final_idx = sorted(set(final_idx))
    
    # 提取關鍵點的座標（從原始路徑中取出）
    key_points = [paths[idx].tolist() for idx in final_idx]
    
    # 過濾過近的關鍵點
    filtered_key_points, filtered_idx = filter_key_points(
        key_points, 
        final_idx, 
        [],  # 不使用max_values進行過濾
        min_distance=20  # 最小距離閾值
    )

    if debug:
        custom_print(ifserver,f"找到 {len(key_points)} 個關鍵點，過濾後剩餘 {len(filtered_key_points)} 個")
        custom_print(ifserver,f"原始路徑中的關鍵點索引: {filtered_idx}")
        custom_print(ifserver,f"關鍵點座標: {filtered_key_points}")

    mapped_idx = find_simplified_indices(paths,filtered_key_points)
    #filtered_key_points=convert_pairs_to_tuples(filtered_key_points)
    return filtered_key_points, mapped_idx
