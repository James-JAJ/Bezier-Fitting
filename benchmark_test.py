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

# ==== 2. 實驗流程主體：支援多指標 ====
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

        for name, metric in metrics.items():
            scores = []
            for level in levels:
                distorted = apply_perturbation(base.copy(), level)
                score = metric(base, distorted)
                scores.append(score)
            results[name].append(scores)

    # ==== 3. 畫圖 ====
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

# ==== 4. 在這裡放入你要測試的指標 ====
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

# ==== 5. 執行 ====
if __name__ == '__main__':
    folder = "benchmarkimg"  # 改成你的圖像資料夾
    font = "benchmarkimg/NotoSansTC-VariableFont_wght.ttf"  # 使用者字體
    metrics_dict = {
        'MSE': mse_similarity,
        'RMSE': rmse_similarity,
        'PSNR': psnr_similarity,
        'SSIM': ssim_similarity,
        'FRSS': frss_shape_similarity,
        
        
        
    }
    evaluate_images_multiple_metrics(folder, metrics=metrics_dict, font_path=font)

