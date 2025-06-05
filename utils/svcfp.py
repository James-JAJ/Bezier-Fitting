import numpy as np
import sys
import os
from .server_tools import *
from .math_tools import *

# 貝茲曲線切割和路徑簡化算法

def distance(p1, p2):
    """
    計算兩點間的歐幾里得距離
    
    Args:
        p1, p2 (array-like): 兩個點的座標
    
    Datatype:
        可以是 list, tuple 或 numpy array
    
    Returns:
        float: 兩點間的距離
    
    ⚠️ 備註:
        - 支援任意維度的點座標計算
        - 內部會自動轉換為 numpy array 進行計算
    """
    return np.linalg.norm(np.array(p1) - np.array(p2))

def perpendicular_distance(point, line_start, line_end):
    """
    計算點到線段的垂直距離
    
    Args:
        point (array-like): 目標點的座標
        line_start (array-like): 線段起點座標
        line_end (array-like): 線段終點座標
    
    Datatype:
        點座標可以是 list, tuple 或 numpy array
    
    Returns:
        float: 點到線段的最短垂直距離
    
    ⚠️ 備註:
        - 如果線段起點和終點相同，返回點到起點的距離
        - 使用投影方法計算真實的垂直距離
        - 考慮投影點在線段範圍內的情況
    """
    if np.array_equal(line_start, line_end):
        return np.linalg.norm(point - line_start)
    
    line_vec = line_end - line_start
    point_vec = point - line_start
    
    line_len = np.dot(line_vec, line_vec)
    t = max(0, min(1, np.dot(point_vec, line_vec) / line_len))
    projection = line_start + t * line_vec
    return np.linalg.norm(point - projection)

def rdp(points, epsilon):
    """
    Douglas-Peucker 道格拉斯-普克線簡化演算法
    
    Args:
        points (array-like): 原始路徑點序列
        epsilon (float): 簡化的容差值，越大簡化程度越高
    
    Datatype:
        points: list of lists/tuples 或 numpy array
        epsilon: float
    
    Returns:
        list: 簡化後的點序列
    
    ⚠️ 備註:
        - 遞歸算法，保留重要的轉折點
        - epsilon 值需要根據座標系統和精度需求調整
        - 點數少於3個時直接返回原序列
        - 返回格式為 list of lists
    """
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

def calculate_angle_change(p1, p2, p3):
    """
    計算三點間的夾角變化
    
    Args:
        p1 (array-like): 第一個點座標
        p2 (array-like): 中間點座標（頂點）
        p3 (array-like): 第三個點座標
    
    Datatype:
        點座標可以是 list, tuple 或 numpy array
    
    Returns:
        float: 夾角度數 (0-180度)
    
    ⚠️ 備註:
        - 計算 p1-p2-p3 三點形成的夾角
        - 返回值越小代表轉角越大
        - 使用單位向量和反餘弦函數計算
        - 自動處理重複點和零向量情況
    """
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
    
    return angle_deg

def svcfp(paths, min_radius=10, max_radius=50, curvature_threshold=27, rdp_epsilon=2, 
          insert_threshold=400, fuse_radio=5, fuse_threshold=10, ifserver=1):
    """
    SVCFP - 智能向量曲線特徵點提取算法
    (Smart Vector Curve Feature Points extraction algorithm)
    
    Args:
        paths (array-like): 原始路徑點序列
        min_radius (int): 最小分析半徑，預設 10
        max_radius (int): 最大分析半徑，預設 50
        curvature_threshold (float): 曲率閾值，預設 27
        rdp_epsilon (float): RDP 簡化容差，預設 2
        insert_threshold (int): 插入中點的距離閾值，預設 400
        fuse_radio (int): 融合半徑，預設 5
        fuse_threshold (float): 融合距離閾值，預設 10
        ifserver (int): 是否為服務器模式（影響輸出），預設 1
    
    Datatype:
        paths: list of lists/tuples 或 numpy array
        其他參數: int/float
    
    Returns:
        tuple: (key_points, final_idx)
            - key_points (list): 提取的關鍵特徵點座標
            - final_idx (list): 關鍵點在原始路徑中的索引
    
    ⚠️ 備註:
        - 結合 RDP 簡化和多尺度特徵分析
        - 自動檢測方向變化和曲率變化點
        - 支援長距離路段的中點插入
        - 包含相近點融合機制
        - 適用於手寫軌跡、路徑規劃等場景
        - 參數需要根據具體應用場景調整
    
    Algorithm Details:
        1. 使用 RDP 算法進行初步簡化
        2. 多尺度半徑分析計算特徵值
        3. 結合角度變化和方向變化檢測
        4. 長距離段落中點插入
        5. 相近特徵點融合優化
    """
    paths = np.array(paths)
    simplified_points = rdp(paths, rdp_epsilon)
    custom_print(ifserver, f"RDP 簡化後的點數: {len(simplified_points)}")

    # 建立簡化點與原始點的索引對應關係
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

    # 內部輔助函數：計算外積符號
    def cross_sign(p1, p2, p3):
        """計算三點外積的符號，用於判斷轉向方向"""
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)
        v1 = p2 - p1
        v2 = p3 - p2
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        return np.sign(cross)

    # 特徵值計算
    stdlist, max_values, angle = [], [], []
    all_feature_values = []
    length = len(paths)

    for i in range(len(simplified_points)):
        if i >= len(original_indices):
            break

        original_idx = original_indices[i]
        angle_change = 0
        cross_sign_change = False

        # 檢測方向變化
        if i > 1 and i < len(simplified_points) - 1:
            A = simplified_points[i - 2]
            B = simplified_points[i - 1]
            C = simplified_points[i]
            D = simplified_points[i + 1]

            sign1 = cross_sign(A, B, C)
            sign2 = cross_sign(B, C, D)

            if sign1 != 0 and sign2 != 0 and sign1 != sign2:
                cross_sign_change = True

        # 計算角度變化
        if i > 0 and i < len(simplified_points) - 1:
            A = simplified_points[i - 1]
            B = simplified_points[i]
            C = simplified_points[i + 1]
            angle_change = calculate_angle_change(A, B, C)
            angle.append(angle_change)
        else:
            angle.append(0)

        # 多尺度半徑分析
        std_values = []
        max_distances = []
        for step_size in range(min_radius, max_radius):
            temp = [paths[original_idx]]

            # 向右擴展
            for k in range(1, step_size + 1):
                right_idx = original_idx + k
                if right_idx < length:
                    temp.append(paths[right_idx])
                else:
                    break

            # 向左擴展
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

        # 計算綜合特徵值
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

    # 提取候選斷點
    candidate_breakpoints = [i for i, val in enumerate(max_values) if val > curvature_threshold]

    if len(simplified_points) > 0:
        if 0 not in candidate_breakpoints:
            candidate_breakpoints.insert(0, 0)
        if len(simplified_points) - 1 not in candidate_breakpoints:
            candidate_breakpoints.append(len(simplified_points) - 1)

    # 長距離段落處理和方向變化檢測
    extended_breakpoints = []
    for i in range(len(candidate_breakpoints) - 1):
        idx1 = original_indices[candidate_breakpoints[i]]
        idx2 = original_indices[candidate_breakpoints[i + 1]]
        extended_breakpoints.append(candidate_breakpoints[i])

        if abs(idx2 - idx1) > insert_threshold:
            # 中點插入
            mid_idx = (idx1 + idx2) // 2
            nearest_idx = np.argmin(np.abs(np.array(original_indices) - mid_idx))
            extended_breakpoints.append(nearest_idx)

            # 尋找方向變化點
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

    # 相近點融合函數
    def fuse_nearby(path, center_idx, radius=fuse_radio, threshold=fuse_threshold):
        """融合相近的特徵點，返回最佳代表點索引"""
        center = path[center_idx]
        indices = [i for i in range(max(0, center_idx-radius), min(len(path), center_idx+radius+1))
                   if np.linalg.norm(path[i] - center) < threshold]
        return center_idx if not indices else indices[len(indices)//2]

    # 修正：生成最終關鍵點 - 確保索引遞增
    final_idx = []
    last_used_idx = -1  # 記錄上一次使用的索引
    
    for i, simplified_idx in enumerate(extended_breakpoints):
        if simplified_idx >= len(simplified_points) or simplified_idx >= len(original_indices):
            continue
            
        # 獲取對應的原始索引
        original_idx = original_indices[simplified_idx]
        
        if i == 0 or i == len(extended_breakpoints) - 1:
            # 保留首尾點，但確保遞增
            if original_idx > last_used_idx:
                final_idx.append(original_idx)
                last_used_idx = original_idx
        else:
            # 其他點進行融合，但確保遞增
            fused_idx = fuse_nearby(paths, original_idx)
            if fused_idx > last_used_idx:
                final_idx.append(fused_idx)
                last_used_idx = fused_idx
            elif original_idx > last_used_idx:
                # 如果融合後的索引不符合遞增要求，使用原始索引
                final_idx.append(original_idx)
                last_used_idx = original_idx
            # 如果都不符合遞增要求，則跳過該點

    # 確保最終結果確實遞增（雙重檢查）
    final_idx_sorted = []
    for idx in final_idx:
        if not final_idx_sorted or idx > final_idx_sorted[-1]:
            final_idx_sorted.append(idx)
    
    final_idx = final_idx_sorted
    key_points = [paths[idx] for idx in final_idx]

    custom_print(ifserver, f"找到 {len(key_points)} 個關鍵點")
    custom_print(ifserver, f"關鍵點原始 index: {final_idx}")
    custom_print(ifserver, f"關鍵點座標: {key_points}")

    return key_points, final_idx