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
def svcfp(paths, min_radius=10, max_radius=50, curvature_threshold=27, rdp_epsilon=20):
    """
    改進版路徑簡化和關鍵點提取演算法，解決偽關鍵點問題
    
    步驟：
    1. 使用 rdp 演算法簡化整個路徑
    2. 將簡化後的點映射回原始路徑的索引
    3. 計算每個點的曲率特徵值
    4. 使用多種過濾方法去除偽關鍵點
    5. 確保起點和終點被包含
    
    參數:
        paths: 原始路徑點列表
        min_radius, max_radius: 搜索範圍的最小和最大半徑
        curvature_threshold: 判斷關鍵點的閾值
        rdp_epsilon: rdp 演算法的簡化閾值
        angle_threshold: 角度變化閾值(弧度)，用於過濾直線段上的點
    
    返回:
        關鍵點座標列表 [[x1, y1], [x2, y2], ...]
        所有點位的特徵值列表 [value1, value2, ...]
    """
    # 確保 paths 是 numpy 數組
    paths = np.array(paths)
    
    # 首先使用 rdp 簡化整個路徑
    simplified_paths = rdp(paths, rdp_epsilon)
    custom_print(f"rdp 簡化後的點數: {len(simplified_paths)}")
    
    # 將簡化後的點映射回原始路徑的索引
    original_indices = []
    i = 0
    check_paths_idx = 0
    custom_print(simplified_paths[i])
    
    while check_paths_idx < len(paths) and i < len(simplified_paths):
        if np.array_equal(paths[check_paths_idx],simplified_paths[i]):
            original_indices.append(check_paths_idx)
            i += 1  # 只有在找到匹配時才移動到 simplified_paths 的下一個元素
        check_paths_idx += 1 # 總是檢查 paths 的下一個元素

    # 處理 simplified_paths 中剩餘的元素 (如果 paths 提早結束)
    if i < len(simplified_paths):
        custom_print(f"警告: simplified_paths 中剩餘的元素無法在 paths 中按順序找到。")
    custom_print(f"簡化點對應的原始索引: {original_indices}")
    
    # 計算各點的標準差、極值數據和角度變化
    stdlist = []
    max_values = []
    angle = []
    all_feature_values = []  # 存儲所有點的特徵值，用於可視化
    
    # 計算角度變化函數
    def calculate_angle_change(p1, p2, p3):
        if np.array_equal(p1, p2) or np.array_equal(p2, p3):
            return 0
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        # 計算單位向量
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        
        # 計算夾角的餘弦值
        angle = np.dot(v1_norm,v2_norm)
        custom_print(angle)
        # 轉換為角度
        
        return angle    
    
    # 為了計算角度變化，需要在簡化路徑上有三個連續點
    for i in range(len(original_indices)):
        # 獲取簡化後路徑中的當前點
        original_idx = original_indices[i]
        
        # 計算角度變化（如果有前後點）
        angle_change = 0
        if i > 0 and i < len(original_indices) - 1:
            prev_idx = original_indices[i-1]
            next_idx = original_indices[i+1]
            angle_change = calculate_angle_change(paths[prev_idx], paths[original_idx], paths[next_idx])
            angle.append(angle_change) #COS值
        else:
            angle.append(0)
        
        std_values = []
        max_distances = []
        
        for j in range(min_radius, max_radius):
            temp = []
            
            # 確保左右搜尋點位數量一致
            left_count = 0
            right_count = 0
            
            # 向右搜尋
            k = 0
            while original_idx + k < len(paths) and distance(paths[original_idx], paths[original_idx+k]) < j:
                temp.append(paths[original_idx+k])
                right_count += 1
                k += 1
            
            # 向左搜尋
            k = 1  # 從 1 開始避免重複添加當前點
            while original_idx - k >= 0 and distance(paths[original_idx], paths[original_idx-k]) < j:
                temp.append(paths[original_idx-k])
                left_count += 1
                k += 1
            
            # 調整左右搜尋點位數量，使其一致
            diff = abs(left_count - right_count)
            if diff > 0:
                if left_count > right_count:
                    # 向右增加點位
                    k = right_count
                    while original_idx + k < len(paths) and distance(paths[original_idx], paths[original_idx+k]) < j and diff > 0:
                        temp.append(paths[original_idx+k])
                        k += 1
                        diff -= 1
                else:
                    # 向左增加點位
                    k = left_count
                    while original_idx - k >= 0 and distance(paths[original_idx], paths[original_idx-k]) < j and diff > 0:
                        temp.append(paths[original_idx-k])
                        k += 1
                        diff -= 1
            
            if len(temp) < 3:  # 確保有足夠的點進行統計
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
            
            # 修改權重比例，更注重標準差
            combined_value = 0.3 * mean_std + 0.7 * mean_max_dist
            
            # 考慮角度變化

            combined_value *= (2 + angle_change)
            
            stdlist.append(mean_std)
            max_values.append(combined_value)
            
            #custom_print(f"點 {i} (原始索引 {original_idx}): 標準差={mean_std:.2f}, 最大距離={mean_max_dist:.2f}, 角度變化={angle_change:.2f}, 加權值={combined_value:.2f}")
        else:
            # 處理找不到足夠點的情況
            stdlist.append(0)
            max_values.append(0)
            #custom_print(f"點 {i} (原始索引 {original_idx}): 沒有足夠的點進行計算")
        
        # 存儲所有點的特徵值，包括索引和位置信息
        all_feature_values.append({
            'index': i,
            'original_index': original_idx,
            'position': simplified_paths[i],
            'value': max_values[-1] if max_values else 0,
            'angle': angle[-1] if angle else 0
        })
    
    # 尋找加權值超過閾值的點作為候選關鍵點
    candidate_breakpoints = []
    for i in range(len(max_values)):
        if max_values[i] > curvature_threshold:
            candidate_breakpoints.append(i)
            #custom_print(f"候選點 {i} (原始索引 {original_indices[i]}): 加權值={max_values[i]:.2f} > 閾值={curvature_threshold}")

    
    
    # 確保包含起點和終點
    if len(simplified_paths) > 0:
        if 0 not in candidate_breakpoints:
            candidate_breakpoints.insert(0, 0)
            #custom_print("添加起點 0 作為關鍵點")
        if len(simplified_paths) - 1 not in candidate_breakpoints:
            candidate_breakpoints.append(len(simplified_paths) - 1)
            #custom_print(f"添加終點 {len(simplified_paths) - 1} 作為關鍵點")
    #candidate_breakpoints=rdp(candidate_breakpoints,20)
    final_idx = []
    i = 0
    j = 0
    while j < len(paths) and i < len(candidate_breakpoints):
        # 注意：candidate_breakpoints 包含的是簡化路徑中的索引
        # 需要取出簡化路徑中的實際點，然後與原始路徑比較
        cb_idx = candidate_breakpoints[i]
        if np.array_equal(paths[j], simplified_paths[cb_idx]):  # 比較實際座標
            final_idx.append(j)  # 添加原始路徑中的索引
            i += 1  # 移動到下一個候選關鍵點
        j += 1  # 總是檢查 paths 的下一個元素

    # 提取關鍵點的座標（從原始路徑中取出）
    key_points = [paths[idx] for idx in final_idx]

    custom_print(f"找到 {len(key_points)} 個關鍵點")
    custom_print(f"原始路徑中的關鍵點索引: {final_idx}")
    custom_print(f"關鍵點座標: {key_points}")

    # 返回關鍵點座標和原始路徑中的關鍵點索引
    return key_points, final_idx
