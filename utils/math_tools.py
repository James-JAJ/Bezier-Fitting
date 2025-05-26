import numpy as np
from scipy.interpolate import make_interp_spline

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
def make_circular_index(idx, length):
    """ 環狀索引處理 """
    return idx % length

def remove_consecutive_duplicates(array):
    """ 移除連續重複項 """
    if len(array) < 2:
        return array
    result = [array[0]]
    for i in range(1, len(array)):
        if not np.array_equal(array[i], array[i-1]):
            result.append(array[i])
    # 若首尾相同，也移除尾
    if len(result) > 1 and np.array_equal(result[0], result[-1]):
        result.pop()
    return result
def shrink_contours(contours, shrink_factor):
    """將輪廓座標縮小"""
    shrunk = []
    for contour in contours:
        new_contour = np.array(contour * shrink_factor, dtype=np.int32)
        shrunk.append(new_contour)
    return shrunk
def find_simplified_indices(paths, simplified_points):
    indices = []
    for sp in simplified_points:
        found = False
        for i, p in enumerate(paths):
            if np.array_equal(p, sp):
                indices.append(i)
                found = True
                break
        if not found:
            raise ValueError(f"Point {sp} not found in paths.")
    return indices
def convert_pairs_to_tuples(obj):
    if isinstance(obj, list):
        # 如果是長度為2的純數字list → 轉成tuple
        if len(obj) == 2 and all(isinstance(i, (int, float)) for i in obj):
            return tuple(obj)
        # 否則遞迴處理內部
        return [convert_pairs_to_tuples(item) for item in obj]
    return obj  # 若不是list就原樣返回
def chord_length_parameterize(points: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.insert(np.cumsum(distances), 0, 0)

    if cumulative[-1] == 0:
        # 所有點重合，無法參數化，直接均勻分布
        t = np.linspace(0, 1, len(points))
    else:
        t = cumulative / cumulative[-1]  # Normalize to [0, 1]
    
    return t

def fit_fixed_end_bezier(points):
    """
    給定首尾點 P0, P3 與中間曲線點序列 points，擬合中間兩個控制點 P1, P2。
    返回 4 個控制點的 ndarray。
    """
    n = len(points)
    if n < 2:
        return None  # 無法擬合
    P0=points[0]
    P3=points[-1]
    # Chord-length parameterization
    dists = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    cumulative = np.insert(np.cumsum(dists), 0, 0)
    if cumulative[-1] == 0:
        return None
    t = cumulative / cumulative[-1]

    # Bernstein basis (只保留 P1, P2 的基底係數)
    A = np.zeros((n, 2))
    for i in range(n):
        ti = t[i]
        A[i, 0] = 3 * (1 - ti)**2 * ti   # 對應 P1
        A[i, 1] = 3 * (1 - ti) * ti**2   # 對應 P2

    # 右側向量 b = Q_i - (1-t)^3 * P0 - t^3 * P3
    b = points - np.outer((1 - t) ** 3, P0) - np.outer(t ** 3, P3)

    # 解最小二乘: Ax = b → x ≈ [P1, P2]
    try:
        Px, _, _, _ = np.linalg.lstsq(A, b[:, 0], rcond=None)
        Py, _, _, _ = np.linalg.lstsq(A, b[:, 1], rcond=None)
    except np.linalg.LinAlgError:
        return None

    P1 = np.array([Px[0], Py[0]])
    P2 = np.array([Px[1], Py[1]])

    #print([P0,P1,P2,P3])
    return [tuple(P0), tuple(P1), tuple(P2), tuple(P3)]

def fit_least_squares_bezier(points):
    """
    最小平方法擬合三階貝茲曲線，首尾控制點固定，求出中間兩個控制點。
    :param points: 軌跡點列，形如 [(x0,y0), (x1,y1), ..., (xn,yn)]
    :return: 4 個控制點 [P0, P1, P2, P3]
    """
    points = np.array(points)
    n = len(points)
    if n < 4:
        raise ValueError("至少需要4個點進行擬合")

    P0 = points[0]
    P3 = points[-1]

    # 參數化 t 值（均勻分佈）
    t = np.linspace(0, 1, n)
    B1 = 3 * (1 - t)**2 * t
    B2 = 3 * (1 - t) * t**2

    # 應變量 Y：扣掉固定點貢獻
    C = points - np.outer((1 - t)**3, P0) - np.outer(t**3, P3)

    # 組成線性系統 A * [P1; P2] = C
    A = np.vstack([B1, B2]).T  # n x 2
    # 解兩個線性系統，X 和 Y 分開解
    AT_A = A.T @ A
    AT_Cx = A.T @ C[:, 0]
    AT_Cy = A.T @ C[:, 1]

    Px = np.linalg.solve(AT_A, AT_Cx)
    Py = np.linalg.solve(AT_A, AT_Cy)

    P1 = np.array([Px[0], Py[0]])
    P2 = np.array([Px[1], Py[1]])

    return [P0, P1, P2, P3]
#垃圾
def fit_fixed_end_bspline(points):
    """
    使用三次 B-spline 擬合點列，固定首尾點，回傳 4 組控制點
    """
    points = np.array(points)
    n = len(points)
    if n < 4:
        return [tuple(points[0])] * 4  # 資料太少，退回平線

    # 首尾固定
    P0 = points[0]
    P3 = points[-1]

    # 建立中間樣本點
    t = np.linspace(0, 1, n)
    x = points[:, 0]
    y = points[:, 1]

    # 使用 scipy 的 make_interp_spline 擬合樣條
    try:
        spline_x = make_interp_spline(t, x, k=3)
        spline_y = make_interp_spline(t, y, k=3)

        # 從 spline 的 t 值中取出中段控制點
        control_t = [1/3, 2/3]
        P1 = np.array([spline_x(control_t[0]), spline_y(control_t[0])])
        P2 = np.array([spline_x(control_t[1]), spline_y(control_t[1])])
    except:
        # 退回為線段
        return [tuple(P0)] * 4

    return [tuple(P0), tuple(P1), tuple(P2), tuple(P3)]