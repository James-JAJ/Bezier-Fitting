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
def svcfp(paths, min_radius=10, max_radius=50, curvature_threshold=27, rdp_epsilon=2,insert_threshold=400,fuse_radio=5,fuse_threshold=10, ifserver=1):
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
        if i >= len(original_indices):
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

    # --- 以下為新增區塊 ---
    extended_breakpoints = []
    for i in range(len(candidate_breakpoints) - 1):
        idx1 = original_indices[candidate_breakpoints[i]]
        idx2 = original_indices[candidate_breakpoints[i + 1]]
        extended_breakpoints.append(candidate_breakpoints[i])

        if abs(idx2 - idx1) > insert_threshold:
            # 中點加入
            mid_idx = (idx1 + idx2) // 2
            nearest_idx = np.argmin(np.abs(np.array(original_indices) - mid_idx))
            extended_breakpoints.append(nearest_idx)

            # 從該段 simplified points 尋找外積轉向變化點
            rdp_range = [j for j in range(candidate_breakpoints[i]+1, candidate_breakpoints[i+1]-1)]
            for k in rdp_range:
                if k+1 >= len(simplified_points):
                    continue
                p1 = simplified_points[k-1]
                p2 = simplified_points[k]
                p3 = simplified_points[k+1]
                sign1 = cross_sign(p1, p2, p3)
                sign2 = cross_sign(p2, p3, p1)
                if sign1 != 0 and sign2 != 0 and sign1 != sign2:
                    extended_breakpoints.append(k)
                    break

    extended_breakpoints.append(candidate_breakpoints[-1])
    extended_breakpoints = sorted(set(extended_breakpoints))

    # 融合相近點（保留首尾）
    def fuse_nearby(path, center_idx, radius=fuse_radio, threshold=fuse_threshold):
        center = path[center_idx]
        indices = [i for i in range(max(0, center_idx-radius), min(len(path), center_idx+radius+1))
                   if np.linalg.norm(path[i] - center) < threshold]
        return center_idx if not indices else indices[len(indices)//2]

    final_idx = []
    for i, idx in enumerate(extended_breakpoints):
        if idx >= len(simplified_points):
            continue
        for j in range(len(paths)):
            if np.array_equal(paths[j], simplified_points[idx]):
                if i == 0 or i == len(extended_breakpoints)-1:
                    final_idx.append(j)
                else:
                    final_idx.append(fuse_nearby(paths, j))
                break

    key_points = [paths[idx] for idx in final_idx]

    custom_print(ifserver, f"找到 {len(key_points)} 個關鍵點")
    custom_print(ifserver, f"關鍵點原始 index: {final_idx}")
    custom_print(ifserver, f"關鍵點座標: {key_points}")

    return key_points, final_idx

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

