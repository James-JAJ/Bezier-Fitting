import numpy as np
from scipy.interpolate import make_interp_spline
import svgwrite
from scipy.spatial import procrustes
from scipy.spatial import KDTree
import cv2
from utils import *

def distance(p1, p2):
    """ 計算兩點間的歐幾里得距離
    Args:
        p1, p2 (array-like): 兩個點的座標
        Datatype: 可以是 list, tuple 或 numpy array
    Returns:
        float: 兩點間的距離
    ⚠️ 備註:
    - 支援任意維度的點座標計算
    - 內部會自動轉換為 numpy array 進行計算
    """
    return np.linalg.norm(np.array(p1) - np.array(p2))
def find_common_elements(arr1, arr2):
    """ 找出兩個陣列中的相同元素
    Args:
        arr1, arr2 (array-like): 兩個要比較的陣列
    Returns:
        numpy.ndarray: 包含相同元素的陣列
    ⚠️ 備註:
    - 使用 numpy.intersect1d，會自動去重並排序
    - 回傳的是 numpy array 格式
    """
    return np.intersect1d(arr1, arr2)
def remove_duplicates(arr):
    
    seen = set()
    result = []
    for item in arr:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
def remove_close_points(path, points, first_point, last_point, threshold):
    """ 在路徑中移除相近的點，但保留首尾點
    Args:
        path (list): 完整路徑點列表
        points (list): 要篩選的點列表
        first_point, last_point (tuple): 首尾點座標
        threshold (int): 距離閾值（以路徑索引為單位）
    Returns:
        list: 篩選後的點列表，包含首尾點
    ⚠️ 備註:
    - threshold 是以路徑中點的索引距離為單位，非歐幾里得距離
    - 會自動處理倒數第二點與終點過近的情況
    - 首尾點一定會被保留
    """
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
def add_mid_points(path, rivise_points, threshold):
    """ 在兩點間距離過大時添加中間點
    Args:
        path (list): 完整路徑點列表
        rivise_points (list): 要修正的點列表（會被直接修改）
        threshold (int): 距離閾值，超過此值會插入中間點
    Returns:
        list: 添加中間點後的點列表
    ⚠️ 備註:
    - 會直接修改傳入的 rivise_points 列表
    - 使用等分法在兩點間插入適當數量的中間點
    - 最終會按照原路徑順序重新排序
    """
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
def mean_min_dist(A, B):
        """ 計算兩個點集合之間的平均最小距離
        Args:
            A, B (numpy.ndarray): 兩個點集合，形狀為 (n, d)
        Returns:
            float: 平均最小距離
        ⚠️ 備註:
        - 對於集合 B 中的每個點，找到 A 中最近點的距離
        - 回傳所有這些最小距離的平均值
        - A 和 B 應該是相同維度的點集合
        """
        return np.mean([np.min(np.linalg.norm(A - b, axis=1)) for b in B])
def interpolate_points(points, step=1):
    """ 使用線性插值來補足缺失的點，使路徑更加平滑
    Args:
        points (list of tuple): 原始離散點列表 [(x1, y1), (x2, y2), ...]
        step (float): 插值間距，數值越小補的點越多，預設為 1
    Returns:
        list of tuple: 平滑後的點列表
    ⚠️ 備註:
    - 會根據兩點間的距離自動決定插值數量
    - 回傳的座標會轉換為整數格式
    - step 參數控制插值密度，建議根據影像解析度調整
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
    """ 環狀索引處理函數
    Args:
        idx (int): 原始索引
        length (int): 陣列長度
    Returns:
        int: 處理後的有效索引
    ⚠️ 備註:
    - 用於處理環狀結構，當索引超出範圍時會自動回到開頭
    - 支援負數索引
    """
    return idx % length
def remove_consecutive_duplicates(array):
    """ 移除陣列中連續重複的項目
    Args:
        array (list): 輸入陣列
    Returns:
        list: 移除連續重複項後的陣列
    ⚠️ 備註:
    - 只移除連續重複的項目，非連續的重複項會保留
    - 如果首尾元素相同，會移除尾部元素（適用於封閉路徑）
    - 使用 numpy.array_equal 進行比較，適用於多維陣列
    """
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
    """ 將輪廓座標按比例縮小
    Args:
        contours (list): 輪廓列表，每個輪廓為座標陣列
        shrink_factor (float): 縮放係數
    Returns:
        list: 縮放後的輪廓列表
    ⚠️ 備註:
    - 所有座標會乘以 shrink_factor 後轉為 int32 格式
    - 適用於 OpenCV 輪廓格式
    """
    shrunk = []
    for contour in contours:
        new_contour = np.array(contour * shrink_factor, dtype=np.int32)
        shrunk.append(new_contour)
    return shrunk
def find_simplified_indices(paths, simplified_points):
    """ 在路徑中找到簡化點的對應索引
    Args:
        paths (list): 完整路徑點列表
        simplified_points (list): 簡化後的點列表
    Returns:
        list: 對應的索引列表
    Raises:
        ValueError: 當簡化點在路徑中找不到時
    ⚠️ 備註:
    - 使用精確匹配查找，要求點座標完全相同
    - 如果找不到對應點會拋出異常
    """
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
    """ 將巢狀列表中的數字對轉換為元組
    Args:
        obj: 任意巢狀結構的物件
    Returns:
        轉換後的物件，數字對會變成元組
    ⚠️ 備註:
    - 遞迴處理巢狀結構
    - 只轉換長度為 2 且全為數字的列表
    - 其他格式的資料保持不變
    """
    if isinstance(obj, list):
        # 如果是長度為2的純數字list → 轉成tuple
        if len(obj) == 2 and all(isinstance(i, (int, float)) for i in obj):
            return tuple(obj)
        # 否則遞迴處理內部
        return [convert_pairs_to_tuples(item) for item in obj]
    return obj  # 若不是list就原樣返回
def chord_length_parameterize(points: np.ndarray) -> np.ndarray:
    """ 使用弦長參數化方法為點序列生成參數
    Args:
        points (numpy.ndarray): 點座標陣列，形狀為 (n, 2)
    Returns:
        numpy.ndarray: 參數化後的 t 值，範圍 [0, 1]
    ⚠️ 備註:
    - 基於相鄰點間的累積距離進行參數化
    - 如果所有點重合，會使用均勻分布作為備案
    - 常用於曲線擬合的前處理步驟
    """
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.insert(np.cumsum(distances), 0, 0)

    if cumulative[-1] == 0:
        # 所有點重合，無法參數化，直接均勻分布
        t = np.linspace(0, 1, len(points))
    else:
        t = cumulative / cumulative[-1]  # Normalize to [0, 1]
    
    return t
def fit_fixed_end_bezier(points):
    """ 給定首尾點，擬合中間兩個控制點的三次貝茲曲線
    Args:
        points (list): 目標擬合的點序列
    Returns:
        list of tuple: 四個控制點 [(P0), (P1), (P2), (P3)] 或 None
    ⚠️ 備註:
    - 首尾控制點固定為輸入點的首尾點
    - 使用最小二乘法求解中間兩個控制點
    - 點數不足時會回傳 None
    - 使用弦長參數化提升擬合品質
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
    """ 最小平方法擬合三階貝茲曲線，首尾控制點固定
    Args:
        points (list): 目標擬合的點序列
    Returns:
        list of tuple: 四個控制點 [(P0), (P1), (P2), (P3)]
    ⚠️ 備註:
    - 若點數不足 4 點，會使用簡化方法估計中間控制點
    - 2 點時：中間控制點取線段的 1/3 和 2/3 處
    - 3 點時：兩個中間控制點都設為中間點
    - 4 點以上：使用完整的最小二乘法
    """
    points = np.array(points)
    n = len(points)

    P0 = points[0]
    P3 = points[-1]

    if n < 4:
        if n == 2:
            # 僅起點終點，中間點取線段內部分佈
            P1 = P0 + (P3 - P0) * 1 / 3
            P2 = P0 + (P3 - P0) * 2 / 3
        elif n == 3:
            P1 = points[1]
            P2 = points[1]
        else:
            raise ValueError("點數過少，無法擬合")

        return [tuple(P0), tuple(P1), tuple(P2), tuple(P3)]

    # 否則執行最小平方法
    t = np.linspace(0, 1, n)
    B1 = 3 * (1 - t) ** 2 * t
    B2 = 3 * (1 - t) * t ** 2
    C = points - np.outer((1 - t) ** 3, P0) - np.outer(t ** 3, P3)

    A = np.vstack([B1, B2]).T
    AT_A = A.T @ A
    AT_Cx = A.T @ C[:, 0]
    AT_Cy = A.T @ C[:, 1]

    Px = np.linalg.solve(AT_A, AT_Cx)
    Py = np.linalg.solve(AT_A, AT_Cy)

    P1 = np.array([Px[0], Py[0]])
    P2 = np.array([Px[1], Py[1]])

    return [tuple(P0), tuple(P1), tuple(P2), tuple(P3)]
def fit_fixed_end_bspline(points):
    """ 使用三次 B-spline 擬合點列，固定首尾點
    Args:
        points (list): 目標擬合的點序列
    Returns:
        list of tuple: 四個控制點 [(P0), (P1), (P2), (P3)]
    ⚠️ 備註:
    - 點數不足 4 時會退回到平線（所有控制點相同）
    - 使用 scipy 的 make_interp_spline 進行樣條擬合
    - 發生異常時會退回到平線處理
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

def nss_interpolate_path(path, num_points=500):
    """使用線性插值強制重取 num_points 點（含起點與終點）"""
    path = np.array(path, dtype=np.float32)

    # 累積距離做參數化
    distances = np.cumsum([0] + [np.linalg.norm(path[i] - path[i - 1]) for i in range(1, len(path))])
    total_length = distances[-1]
    if total_length == 0:
        return np.tile(path[0], (num_points, 1))  # 所有點相同

    distances /= total_length  # 正規化到 0~1
    target_positions = np.linspace(0, 1, num_points)

    # 線性插值 x/y 分開處理
    x_interp = np.interp(target_positions, distances, path[:, 0])
    y_interp = np.interp(target_positions, distances, path[:, 1])
    return np.stack([x_interp, y_interp], axis=1)
def nss_normalized_shape_similarity(path1, path2, num_points=500):
    path1 = nss_interpolate_path(path1, num_points)
    path2 = nss_interpolate_path(path2, num_points)

    def normalize_path(path):
        center = np.mean(path, axis=0)
        centered = path - center
        scale = np.max(np.linalg.norm(centered, axis=1))
        return centered / scale if scale > 0 else centered

    path1 = normalize_path(path1)
    path2 = normalize_path(path2)

    distances = np.linalg.norm(path1 - path2, axis=1)
    mean_distance = np.mean(distances)
    similarity_score = 1 - mean_distance
    return similarity_score

def gss_shape_similarity(points1, points2):
    """
    幾何不變形狀相似度 (GSS)
    比較兩組輪廓點集（無序），不受平移、旋轉與尺度影響
    回傳值越小越相似（0為完全一致）

    Parameters:
        points1, points2: np.ndarray (Nx2)
            輪廓點集合

    Returns:
        float: Procrustes 損失（越小越相似）
    """
    # 點數需相同，否則先內插統一長度
    def resample(path, num_points=100):
        path = np.array(path, dtype=np.float32)
        dists = np.cumsum([0] + [np.linalg.norm(path[i] - path[i-1]) for i in range(1, len(path))])
        if dists[-1] == 0:
            return np.tile(path[0], (num_points, 1))
        dists /= dists[-1]
        target = np.linspace(0, 1, num_points)
        x = np.interp(target, dists, path[:, 0])
        y = np.interp(target, dists, path[:, 1])
        return np.stack([x, y], axis=1)

    A = resample(points1)
    B = resample(points2)

    # procrustes自帶去平移、尺度與旋轉後計算平均誤差
    mtx1, mtx2, disparity = procrustes(A, B)
    return disparity

def frss_shape_similarity(contours1, contours2, num_points=100):
    from scipy.spatial import KDTree

    def normalize(path):
        path = np.array(path, dtype=np.float32)
        center = np.mean(path, axis=0)
        centered = path - center
        scale = np.max(np.linalg.norm(centered, axis=1))
        return centered / scale if scale > 0 else centered

    def resample(path, num=100):
        if len(path) < 2:
            return np.tile(path[0], (num, 1))
        dists = np.cumsum([0] + [np.linalg.norm(path[i] - path[i - 1]) for i in range(1, len(path))])
        dists /= dists[-1] if dists[-1] != 0 else 1
        target = np.linspace(0, 1, num)
        x = np.interp(target, dists, path[:, 0])
        y = np.interp(target, dists, path[:, 1])
        return np.stack([x, y], axis=1)

    valid1 = [c.squeeze() for c in contours1 if c.shape[0] >= 5]
    valid2 = [c.squeeze() for c in contours2 if c.shape[0] >= 5]
    points1 = np.vstack(valid1) if valid1 else np.zeros((1, 2))
    points2 = np.vstack(valid2) if valid2 else np.zeros((1, 2))

    if len(points1) < 2 or len(points2) < 2:
        return 0.0

    A = normalize(resample(points1, num_points))
    B = normalize(resample(points2, num_points))

    tree_A = KDTree(A)
    tree_B = KDTree(B)
    dists1, _ = tree_B.query(A)
    dists2, _ = tree_A.query(B)

    avg_dist = (np.mean(dists1) + np.mean(dists2)) / 2
    return 1 / (1 + avg_dist)
