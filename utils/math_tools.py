import numpy as np

#distance:距離差
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))
#math
#find_common_elements:找二陣列陣列相同元素
def find_common_elements(arr1, arr2):
    return np.intersect1d(arr1, arr2)
#remove_duplicates:剔除重複元素
def remove_duplicates(arr):
    seen = set()
    result = []
    for item in arr:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
#remove_close_points:在閾值內刪除相近元跳過首尾點，回傳一包含首尾二點新列表
def remove_close_points(path, points, first_point, last_point, threshold):
    if not points:
        return []
    
    path_index = {point: i for i, point in enumerate(path)}

    filtered_points = [first_point]  # 保留首點
    last_kept_index = path_index[first_point]  # 追蹤上一次保留點的索引

    for point in points:
        if point in {first_point, last_point}:  # 跳過首尾點
            continue

        current_index = path_index[point]
        if current_index - last_kept_index >= threshold:
            filtered_points.append(point)
            last_kept_index = current_index  # 更新最後保留的點索引
    
    # 如果倒數第二個點離終點太近，則刪除
    if len(filtered_points) > 2 and path_index[filtered_points[-2]] + threshold >= path_index[last_point]:
        filtered_points.pop(-2)
    
    filtered_points.append(last_point)  # 保留尾點
    return filtered_points
#add_mid_points:在兩點間添加中點
def add_mid_points(path, rivise_points, threshold):
    # 預先建立 path 中點的索引對應表，加快查找速度
    path_index = {point: i for i, point in enumerate(path)}

    new_points = []  # 存放要新增的點

    for i in range(len(rivise_points) - 1):
        idx1 = path_index[rivise_points[i]]
        idx2 = path_index[rivise_points[i + 1]]
        dis = abs(idx1 - idx2)

        if dis >= threshold:
            num_segments = dis // threshold  # 計算應該插入的點數
            for j in range(1, num_segments + 1):
                mid_index = idx1 + (j * dis) // (num_segments + 1)  # 等分索引位置
                new_points.append(path[mid_index])  # 插入該索引對應的點

    # 合併原本的點與新增的點並排序
    rivise_points.extend(new_points)
    rivise_points.sort(key=lambda p: path_index[p])  # 按原本 path 順序排列

    return rivise_points
#mean_min_dist:計算兩個集合之間的平均最小距離
def mean_min_dist(A, B):
        return np.mean([np.min(np.linalg.norm(A - b, axis=1)) for b in B])
#interpolate_points:使用線性插值來補足缺失的點，使路徑更加平滑
def interpolate_points(points, step=1):
    """
    使用線性插值來補足缺失的點，使路徑更加平滑
    :param points: 原始離散點列表 [(x1, y1), (x2, y2), ...]
    :param step: 插值間距，數值越小補的點越多
    :return: 平滑後的點列表
    """
    new_points = []
    
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        
        # 計算兩點之間的距離
        dist = np.linalg.norm([x2 - x1, y2 - y1])
        
        # 根據距離決定插值數量
        num_steps = max(int(dist / step), 1)
        
        for t in np.linspace(0, 1, num_steps):
            new_x = int(x1 + (x2 - x1) * t)
            new_y = int(y1 + (y2 - y1) * t)
            new_points.append((new_x, new_y))
    
    return new_points