import cv2
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import warnings
warnings.filterwarnings('ignore')
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')


import cv2
import numpy as np
from scipy.spatial import procrustes
from scipy.optimize import linear_sum_assignment
from pyemd import emd
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#print(os.getcwd())
from utils import *
# 座標格式轉換器
# 將多層嵌套的座標列表轉換為簡單的 [(),()]格式

def convert_coordinates(nested_coords):

    result = []
    
    def extract_coords(coords):

        if isinstance(coords, list):
            if len(coords) == 2 and all(isinstance(x, (int, float)) for x in coords):
                # 這是一個座標點 [x, y]
                return [tuple(coords)]
            else:
                # 這是一個包含多個元素的列表，遞歸處理
                extracted = []
                for item in coords:
                    extracted.extend(extract_coords(item))
                return extracted
        elif isinstance(coords, tuple) and len(coords) == 2:
            # 這已經是一個座標元組 (x, y)
            return [coords]
        else:
            return []
    
    # 對輸入的每個多邊形進行處理
    for polygon in nested_coords:
        polygon_coords = extract_coords(polygon)
        result.extend(polygon_coords)
    
    return result
# ========== Utility Functions ==========

def load_and_preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea).squeeze(axis=1)  # Return the largest contour

def save_transformation_matrix(matrix, filename):
    np.save(filename, matrix)


# ========== Alignment Methods ==========

def ransac_affine_transform(src, dst):
    matrix, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)
    return matrix


def procrustes_analysis(src, dst):
    mtx1, mtx2, _ = procrustes(src, dst)
    return mtx2  # Transformed dst points aligned to src


def emd_alignment(src, dst):
    def pairwise_dist(p1, p2):
        return np.linalg.norm(p1[:, None] - p2[None, :], axis=2).astype(np.float64)

    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    dist_matrix = pairwise_dist(src, dst)

    # Uniform weights
    src_weight = np.ones(len(src)) / len(src)
    dst_weight = np.ones(len(dst)) / len(dst)

    flow = emd(src_weight.tolist(), dst_weight.tolist(), dist_matrix.tolist())
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    aligned_dst = dst[col_ind]
    return aligned_dst


# ========== Main Comparison Pipeline ==========



# 主程式
if __name__ == "__main__":
    # --- 可調參數 ---
    img1_path = "Strike_Analyzing/target.png"
    img2_path = "Strike_Analyzing/test2.png"
    scale_factor = 2             # 前處理放大倍數
    final_shrink_factor = 0.5    # 縮小倍數
    blur_ksize = 3               # 模糊核大小  
    threshold_value = 180        # 二質化閾值
    epsilon = 1.0                # 簡化輪廓的誤差
    rdp_epsilon = 3              # RDP簡化閾值
    curvature_threshold = 30     # 曲率閾值
    min_radius = 10              # 最小搜尋半徑
    max_radius = 50              # 最大搜尋半徑
    debug = True                 # 是否打印除錯信息
    ifshow = 0                   # 是否中途顯示
    # ----------------

    try:
        contour1 = load_and_preprocess_image(img1_path)
        contour2 = load_and_preprocess_image(img2_path)

        # 原圖 灰階圖
        original_img1, gray_img1 = inputimg_colortogray(img1_path)
        original_img2, gray_img2 = inputimg_colortogray(img2_path)

        # 前處理圖片
        preprocessed_img1 = preprocess_image(gray_img1, scale_factor, blur_ksize, threshold_value, ifshow)
        preprocessed_img2 = preprocess_image(gray_img2, scale_factor, blur_ksize, threshold_value, ifshow)

        # 得到圖片輪廓
        contours1 = getContours(preprocessed_img1, ifshow)
        contours2 = getContours(preprocessed_img2, ifshow)

        svcfplist1 = []
        svcfplist2 = []
        # 處理每個輪廓
        for contour in contours1:
            
            fixcontour = [sublist[0] for sublist in contour]
            fixcontour = remove_consecutive_duplicates(fixcontour)  # 移除首尾或相鄰重複點
            rdp_points = rdp(fixcontour, epsilon=rdp_epsilon)
            print("RDP簡化後的點數:", len(rdp_points))

            custom_points, custom_idx = svcfp_queue(
                fixcontour,
                rdp_points,
                min_radius=min_radius,
                max_radius=max_radius,
                curvature_threshold=curvature_threshold,
                debug=debug,
                ifserver=0
            )
            svcfplist1.append(custom_points)
        for contour in contours2:
            
            fixcontour = [sublist[0] for sublist in contour]
            fixcontour = remove_consecutive_duplicates(fixcontour)  # 移除首尾或相鄰重複點
            rdp_points = rdp(fixcontour, epsilon=rdp_epsilon)
            print("RDP簡化後的點數:", len(rdp_points))

            custom_points, custom_idx = svcfp_queue(
                fixcontour,
                rdp_points,
                min_radius=min_radius,
                max_radius=max_radius,
                curvature_threshold=curvature_threshold,
                debug=debug,
                ifserver=0
            )
            svcfplist2.append(custom_points)


        keypoints1 = np.array(convert_coordinates(svcfplist1))
        keypoints2 = np.array(convert_coordinates(svcfplist2))

        cov=covariance_between_point_sets(keypoints1,keypoints2)
        print(cov)

        
    except Exception as e:
        print(f"發生錯誤: {e}")
        import traceback
        traceback.print_exc()
