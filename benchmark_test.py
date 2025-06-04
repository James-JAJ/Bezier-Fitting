import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib import font_manager
from typing import Callable, Dict
from skimage.metrics import structural_similarity as ssim
from utils import *  # 導入所有工具函數，包括 server_tools 中的函數

# ==== 1. 擾動函數（旋轉 + 縮放 + 平移 + 模糊） ====
def apply_perturbation(image: np.ndarray, level: float) -> np.ndarray:
    angle = level * 20
    scale = 1 + level * 0.3
    tx, ty = int(10 * level), int(10 * level)
    rows, cols = image.shape

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, scale)
    image = cv2.warpAffine(image, M, (cols, rows))
    image = cv2.warpAffine(image, np.float32([[1, 0, tx], [0, 1, ty]]), (cols, rows))
    if level > 0:
        image = gaussian_filter(image, sigma=level * 2)
    return image

from scipy.spatial import cKDTree
import numpy as np

def scs_shape_similarity(contours1, contours2):
    """
    SCS（Symmetric Contour Similarity）：
    接收兩組輪廓列表（OpenCV contours），計算其點雲之平均對稱最短距離的相似度（越大越相似）。
    """

    def contours_to_points(contours):
        if contours is None or len(contours) == 0:
            return np.zeros((1, 2), dtype=np.float32)

        valid = [c.reshape(-1, 2) for c in contours if c.shape[0] >= 5]
        if len(valid) == 0:
            return np.zeros((1, 2), dtype=np.float32)

        return np.concatenate(valid, axis=0)

    def mean_min_distance(A, B):
        tree = cKDTree(A)
        dists, _ = tree.query(B)
        return np.mean(dists)

    def symmetric_similarity(A, B):
        if len(A) < 2 or len(B) < 2:
            return 0.0
        avg_dist = (mean_min_distance(A, B) + mean_min_distance(B, A)) / 2
        return 1 / (1 + avg_dist)

    points1 = contours_to_points(contours1)
    points2 = contours_to_points(contours2)

    return float(symmetric_similarity(points1, points2))

# ==== 3. 實驗流程主體（輪廓合併後比對）====
def evaluate_images_multiple_metrics(
    folder_path: str,
    metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]],
    levels: np.ndarray = np.linspace(0, 1, 10),
    font_path: str = None
):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg'))]
    results = {name: [] for name in metrics}

    for file in image_files:
        path = os.path.join(folder_path, file)
        base = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        base = cv2.resize(base, (256, 256))

        base_blur = cv2.GaussianBlur(base, (3, 3), 0)
        _, base_thresh = cv2.threshold(base_blur, 127, 255, cv2.THRESH_BINARY_INV)
        contours_base, _ = cv2.findContours(base_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        contour_list_base = [cnt.squeeze() for cnt in contours_base if cnt.shape[0] >= 5]
        contour_base = np.concatenate(contour_list_base, axis=0) if contour_list_base else np.zeros((1, 2), dtype=np.float32)

        for name, metric in metrics.items():
            scores = []
            for level in levels:
                distorted = apply_perturbation(base.copy(), level)

                if name == 'FRSS':
                    blur = cv2.GaussianBlur(distorted, (3, 3), 0)
                    _, threshed = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)
                    contours_dist, _ = cv2.findContours(threshed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                    contour_list_dist = [cnt.squeeze() for cnt in contours_dist if cnt.shape[0] >= 5]
                    contour_dist = np.concatenate(contour_list_dist, axis=0) if contour_list_dist else np.zeros((1, 2), dtype=np.float32)

                    score = scs_shape_similarity(contour_base, contour_dist)
                else:
                    score = metric(base, distorted)

                scores.append(score)
            results[name].append(scores)

    # ==== 4. 畫圖 ====
    plt.figure(figsize=(10, 5), dpi=100)
    for name, score_matrix in results.items():
        score_array = np.array(score_matrix)
        means = np.mean(score_array, axis=0)
        stds = np.std(score_array, axis=0)
        plt.errorbar(levels, means, yerr=stds, fmt='o-', capsize=3, label=name)

    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.xlabel("擾動程度", fontproperties=font_manager.FontProperties(fname=font_path) if font_path else None)
    plt.ylabel("相似度", fontproperties=font_manager.FontProperties(fname=font_path) if font_path else None)
    plt.title("多指標相似度 ± 標準差（誤差線表示）", fontproperties=font_manager.FontProperties(fname=font_path) if font_path else None)
    plt.xticks(fontproperties=font_manager.FontProperties(fname=font_path) if font_path else None)
    plt.yticks(fontproperties=font_manager.FontProperties(fname=font_path) if font_path else None)
    plt.legend(prop=font_manager.FontProperties(fname=font_path) if font_path else None)
    plt.tight_layout()
    plt.show()

# ==== 5. 指標函數 ====
def ssim_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    return ssim(img1, img2)

def mse_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    err = np.mean((img1.astype("float") - img2.astype("float")) ** 2)
    return 1 / (1 + err)

def rmse_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1.astype("float") - img2.astype("float")) ** 2)
    rmse = np.sqrt(mse)
    return 1 / (1 + rmse)

def psnr_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1.astype("float") - img2.astype("float")) ** 2)
    if mse == 0:
        return 1.0
    PIXEL_MAX = 255.0
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return psnr / 100  # normalize to [0, 1] approx

# ==== 6. 執行 ====
if __name__ == '__main__':
    folder = "benchmarkimg"  # 圖像資料夾
    font = "benchmarkimg/NotoSansTC-VariableFont_wght.ttf"  # 字體檔
    metrics_dict = {
        'MSE': mse_similarity,
        'RMSE': rmse_similarity,
        'PSNR': psnr_similarity,
        'SSIM': ssim_similarity,
        'FRSS': scs_shape_similarity,
    }
    evaluate_images_multiple_metrics(folder, metrics=metrics_dict, font_path=font)
