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

# ==== 2. FRSS 輪廓比對函數（合併所有輪廓點）====
def frss_shape_similarity(points1, points2, num_points=100):
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

    A = normalize(resample(points1, num_points))
    B = normalize(resample(points2, num_points))

    tree_A = KDTree(A)
    tree_B = KDTree(B)
    dists1, _ = tree_B.query(A)
    dists2, _ = tree_A.query(B)

    avg_dist = (np.mean(dists1) + np.mean(dists2)) / 2
    return 1 / (1 + avg_dist)

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

                    score = frss_shape_similarity(contour_base, contour_dist)
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
        'FRSS': frss_shape_similarity,
    }
    evaluate_images_multiple_metrics(folder, metrics=metrics_dict, font_path=font)
