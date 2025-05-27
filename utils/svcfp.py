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
def svcfp(paths, min_radius=10, max_radius=50, curvature_threshold=27, rdp_epsilon=2,insert_threshold=400,insert_angle_threshold=10, ifserver=1):
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

    def cross_sign(p1, p2, p3):
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)
        v1 = p2 - p1
        v2 = p3 - p2
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        return np.sign(cross)

    length = len(paths)

    for i in range(len(simplified_points)):
        if i >= len(original_indices):  # 防呆處理
            break

        original_idx = original_indices[i]
        angle_change = 0
        cross_sign_change = False

        if i > 1 and i < len(simplified_points) - 1:
            A = simplified_points[i - 2]
            B = simplified_points[i - 1]
            C = simplified_points[i]
            D = simplified_points[i + 1]

            sign1 = cross_sign(A, B, C)
            sign2 = cross_sign(B, C, D)

            if sign1 != 0 and sign2 != 0 and sign1 != sign2:
                cross_sign_change = True

        if i > 0 and i < len(simplified_points) - 1:
            A = simplified_points[i - 1]
            B = simplified_points[i]
            C = simplified_points[i + 1]
            angle_change = calculate_angle_change(A, B, C)
            angle.append(angle_change)
        else:
            angle.append(0)

        std_values = []
        max_distances = []
        for step_size in range(min_radius, max_radius):
            temp = [paths[original_idx]]

            for k in range(1, step_size + 1):
                right_idx = original_idx + k
                if right_idx < length:
                    temp.append(paths[right_idx])
                else:
                    break

            for k in range(1, step_size + 1):
                left_idx = original_idx - k
                if left_idx >= 0:
                    temp.append(paths[left_idx])
                else:
                    break

            if len(temp) < 2:
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
            angle_weight = 0.4
            std_weight = 0.3
            dist_weight = 0.7
            angle_factor = 1.0 + (angle_change / 180.0)
            combined_value = (
                std_weight * mean_std + 
                dist_weight * mean_max_dist + 
                angle_weight * angle_change
            ) * angle_factor

            if cross_sign_change:
                combined_value *= 1.1

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

    # 新增：如果任兩個節點距離超過 400，強制在其中加入中點
    extended_breakpoints = []
    for i in range(len(candidate_breakpoints) - 1):
        idx1 = original_indices[candidate_breakpoints[i]]
        idx2 = original_indices[candidate_breakpoints[i + 1]]
        extended_breakpoints.append(candidate_breakpoints[i])

        if abs(idx2 - idx1) > insert_threshold :
            mid_idx = (idx1 + idx2) // 2
            # 找到最接近 mid_idx 的 simplified_point
            nearest_idx = np.argmin(np.abs(np.array(original_indices) - mid_idx))
            A = simplified_points[nearest_idx - 1]
            B = simplified_points[nearest_idx]
            C = simplified_points[nearest_idx + 1]
            angle_change = calculate_angle_change(A, B, C)
            if angle_change < insert_angle_threshold:   
                extended_breakpoints.append(nearest_idx)

    extended_breakpoints.append(candidate_breakpoints[-1])
    extended_breakpoints = sorted(set(extended_breakpoints))

    final_idx = []
    i = j = 0
    while j < len(paths) and i < len(extended_breakpoints):
        cb_idx = extended_breakpoints[i]
        if cb_idx >= len(simplified_points):
            i += 1
            continue
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
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
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

def svcfp_queue(paths,  min_radius=10, max_radius=50, curvature_threshold=27, 
                        rdp_epsilon=2, insert_threshold=400, insert_angle_threshold=10, debug=False, ifserver=1):
    """
    增強版環狀路徑簡化和關鍵點提取演算法
    
    新增功能：
    1. 內外積判斷 (cross_sign) - 環狀版本
    2. 轉向變化檢測 - 環狀版本  
    3. 距離閾值插入機制 - 環狀版本
    
    參數:
        paths: 原始路徑點列表
        simplified_points: 已簡化的路徑點列表
        min_radius: 搜索範圍的最小格數
        max_radius: 搜索範圍的最大格數
        curvature_threshold: 判斷關鍵點的閾值
        rdp_epsilon: RDP簡化參數
        insert_threshold: 距離插入閾值
        insert_angle_threshold: 插入點角度閾值
        debug: 是否打印詳細除錯信息
        ifserver: 服務器模式標記
    
    返回:
        關鍵點座標列表 [[x1, y1], [x2, y2], ...]
        關鍵點在原始路徑中的索引列表
    """
    paths = np.array(paths)
    rdp_points = rdp(paths, epsilon=rdp_epsilon)
    simplified_points = np.array(rdp_points)
    length = len(paths)
    
    # 自動依據長度調整搜尋格數
    if length <= 200:
        min_radius = max(3, int(min_radius * (length/200)))
        max_radius = max(10, int(max_radius * (length/200)))
    
    # 將簡化後的點映射回原始路徑的索引
    original_indices = find_simplified_indices(paths, simplified_points)
    
    def cross_sign(p1, p2, p3):
        """計算三點的外積符號，判斷轉向"""
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)
        v1 = p2 - p1
        v2 = p3 - p2
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        return np.sign(cross)
    
    def get_circular_point(indices_list, center_idx, offset):
        """環狀獲取點，處理邊界循環"""
        target_idx = (center_idx + offset) % len(indices_list)
        return indices_list[target_idx]
    
    # 計算各點的標準差、極值數據和角度變化
    stdlist = []
    max_values = []
    angle = []
    all_feature_values = []
    
    # 為了計算角度變化和轉向檢測，需要在簡化路徑上有足夠的點
    for i in range(len(original_indices)):
        original_idx = original_indices[i]
        angle_change = 0
        cross_sign_change = False
        
        # 環狀轉向變化檢測 - 檢查前後兩個轉向是否不同
        if len(original_indices) >= 4:  # 需要至少4個點才能做轉向變化檢測
            # 獲取環狀的前後點索引
            prev2_original = original_indices[get_circular_point(range(len(original_indices)), i, -2)]
            prev1_original = original_indices[get_circular_point(range(len(original_indices)), i, -1)]
            curr_original = original_indices[i]
            next1_original = original_indices[get_circular_point(range(len(original_indices)), i, 1)]
            
            # 計算兩段的轉向符號
            sign1 = cross_sign(paths[prev2_original], paths[prev1_original], paths[curr_original])
            sign2 = cross_sign(paths[prev1_original], paths[curr_original], paths[next1_original])
            
            # 檢查轉向變化（忽略零值，即直線段）
            if sign1 != 0 and sign2 != 0 and sign1 != sign2:
                cross_sign_change = True
                if debug:
                    print(f"點 {i} 檢測到轉向變化: {sign1} -> {sign2}")
        
        # 環狀角度變化計算
        if len(original_indices) >= 3:
            prev_original = original_indices[get_circular_point(range(len(original_indices)), i, -1)]
            next_original = original_indices[get_circular_point(range(len(original_indices)), i, 1)]
            angle_change = calculate_angle_change(paths[prev_original], paths[original_idx], paths[next_original])
            angle.append(angle_change)
        else:
            angle.append(0)
        
        std_values = []
        max_distances = []
        
        # 使用環狀索引進行特徵計算
        for step_size in range(min_radius, max_radius):
            temp = []
            # 加入當前點
            temp.append(paths[original_idx])
            
            # 向右走step_size格 (環狀)
            for k in range(1, step_size + 1):
                right_idx = make_circular_index(original_idx + k, length)
                temp.append(paths[right_idx])
            
            # 向左走step_size格 (環狀)
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
            std_weight = 0.2
            dist_weight = 0.8
            
            # 使用角度變化值直接作為加權參數
            angle_factor = 1.0 + (angle_change / 180.0)
            
            combined_value = (
                std_weight * mean_std + 
                dist_weight * mean_max_dist + 
                angle_weight * angle_change
            ) * angle_factor 
            
            # 轉向變化加權 - 如果檢測到轉向變化，增加權重
            if cross_sign_change:
                combined_value *= 1.1
                if debug:
                    print(f"點 {i} 因轉向變化加權: {combined_value/1.1:.2f} -> {combined_value:.2f}")
            
            stdlist.append(mean_std)
            max_values.append(combined_value)
            
            if debug:
                print(f"點 {i} (原始索引 {original_idx}): 標準差={mean_std:.2f}, 最大距離={mean_max_dist:.2f}, " +
                      f"角度變化={angle_change:.2f}, 轉向變化={cross_sign_change}, 加權值={combined_value:.2f}")
        else:
            stdlist.append(0)
            max_values.append(0)
            if debug:
                print(f"點 {i} (原始索引 {original_idx}): 沒有足夠的點進行計算")
        
        # 存儲所有點的特徵值
        all_feature_values.append({
            'index': i,
            'original_index': original_idx,
            'position': paths[original_idx].tolist() if isinstance(paths[original_idx], np.ndarray) else paths[original_idx],
            'value': max_values[-1] if max_values else 0,
            'angle': angle[-1] if angle else 0,
            'cross_sign_change': cross_sign_change
        })
    
    # 尋找加權值超過閾值的點作為候選關鍵點
    candidate_breakpoints = []
    for i in range(len(max_values)):
        if max_values[i] > curvature_threshold:
            candidate_breakpoints.append(i)
            if debug:
                print(f"候選點 {i} (原始索引 {original_indices[i]}): 加權值={max_values[i]:.2f} > 閾值={curvature_threshold}")

    # 環狀路徑通常不需要強制包含起點終點，但如果需要可以取消註解
    # if len(simplified_points) > 0:
    #     if 0 not in candidate_breakpoints:
    #         candidate_breakpoints.insert(0, 0)
    #     if len(simplified_points) - 1 not in candidate_breakpoints:
    #         candidate_breakpoints.append(len(simplified_points) - 1)
    
    # 距離閾值插入機制 - 環狀版本
    extended_breakpoints = []
    candidate_breakpoints = sorted(candidate_breakpoints)
    
    for i in range(len(candidate_breakpoints)):
        current_bp = candidate_breakpoints[i]
        next_bp = candidate_breakpoints[(i + 1) % len(candidate_breakpoints)]  # 環狀處理
        
        extended_breakpoints.append(current_bp)
        
        # 計算兩個候選點之間的距離（考慮環狀特性）
        idx1 = original_indices[current_bp]
        idx2 = original_indices[next_bp]
        
        # 計算環狀距離（選擇較短的路徑）
        forward_dist = (idx2 - idx1) % length
        backward_dist = (idx1 - idx2) % length
        circular_dist = min(forward_dist, backward_dist)
        
        if circular_dist > insert_threshold:
            if debug:
                print(f"檢測到距離過長: 點{current_bp}({idx1}) 到 點{next_bp}({idx2}), 距離={circular_dist}")
            
            # 計算中點位置（環狀）
            if forward_dist <= backward_dist:
                mid_idx = make_circular_index(idx1 + forward_dist // 2, length)
            else:
                mid_idx = make_circular_index(idx1 - backward_dist // 2, length)
            
            # 找到最接近 mid_idx 的 simplified_point
            min_dist = float('inf')
            nearest_idx = -1
            for j, orig_idx in enumerate(original_indices):
                dist_to_mid = min(abs(orig_idx - mid_idx), length - abs(orig_idx - mid_idx))
                if dist_to_mid < min_dist:
                    min_dist = dist_to_mid
                    nearest_idx = j
            
            if nearest_idx != -1 and nearest_idx not in extended_breakpoints:
                # 檢查插入點的角度變化
                if len(original_indices) >= 3:
                    prev_idx_in_simplified = get_circular_point(range(len(original_indices)), nearest_idx, -1)
                    next_idx_in_simplified = get_circular_point(range(len(original_indices)), nearest_idx, 1)
                    
                    prev_original = original_indices[prev_idx_in_simplified]
                    curr_original = original_indices[nearest_idx]
                    next_original = original_indices[next_idx_in_simplified]
                    
                    insert_angle = calculate_angle_change(paths[prev_original], paths[curr_original], paths[next_original])
                    
                    if insert_angle < insert_angle_threshold:
                        extended_breakpoints.append(nearest_idx)
                        if debug:
                            print(f"插入中間點 {nearest_idx} (原始索引 {curr_original}), 角度變化={insert_angle:.2f}")
    
    # 去重並排序
    extended_breakpoints = sorted(set(extended_breakpoints))
    
    # 將候選關鍵點從簡化路徑轉換回原始路徑
    final_idx = [original_indices[bp] for bp in extended_breakpoints]
    final_idx = sorted(set(final_idx))
    
    # 提取關鍵點的座標
    key_points = [paths[idx].tolist() for idx in final_idx]
    
    # 過濾過近的關鍵點（考慮環狀特性）
    filtered_key_points, filtered_idx = filter_key_points_circular(
        key_points, 
        final_idx, 
        length,
        min_distance=20
    )

    if debug:
        custom_print(ifserver, f"找到 {len(key_points)} 個關鍵點，過濾後剩餘 {len(filtered_key_points)} 個")
        custom_print(ifserver, f"原始路徑中的關鍵點索引: {filtered_idx}")
        custom_print(ifserver, f"關鍵點座標: {filtered_key_points}")

    # 將過濾後的關鍵點映射回簡化點索引
    mapped_idx = find_simplified_indices(paths, filtered_key_points)
    
    return filtered_key_points, mapped_idx


def filter_key_points_circular(key_points, indices, path_length, min_distance=20):
    """
    環狀路徑的關鍵點過濾函數
    考慮環狀距離進行過近點的移除
    """
    if len(key_points) <= 2:
        return key_points, indices
    
    filtered_points = []
    filtered_indices = []
    
    # 將點和索引配對並按索引排序
    paired = list(zip(key_points, indices))
    paired.sort(key=lambda x: x[1])
    
    filtered_points.append(paired[0][0])
    filtered_indices.append(paired[0][1])
    
    for i in range(1, len(paired)):
        current_idx = paired[i][1]
        last_kept_idx = filtered_indices[-1]
        
        # 計算環狀距離
        forward_dist = (current_idx - last_kept_idx) % path_length
        backward_dist = (last_kept_idx - current_idx) % path_length
        circular_dist = min(forward_dist, backward_dist)
        
        if circular_dist >= min_distance:
            filtered_points.append(paired[i][0])
            filtered_indices.append(paired[i][1])
    
    # 檢查第一個和最後一個點的距離（環狀）
    if len(filtered_indices) > 2:
        first_idx = filtered_indices[0]
        last_idx = filtered_indices[-1]
        forward_dist = (first_idx - last_idx) % path_length
        backward_dist = (last_idx - first_idx) % path_length
        circular_dist = min(forward_dist, backward_dist)
        
        if circular_dist < min_distance:
            # 移除最後一個點
            filtered_points.pop()
            filtered_indices.pop()
    
    return filtered_points, filtered_indices


def make_circular_index(idx, length):
    """環狀索引處理"""
    return idx % length
