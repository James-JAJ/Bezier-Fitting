import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib import font_manager
from typing import Callable, Dict
from skimage.metrics import structural_similarity as ssim
from utils import *              # 含 inputimg_colortogray 等

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

# ==== 2. 主流程 ====
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
        if base is None:
            continue
        base = cv2.resize(base, (256, 256))

        # 預處理基準圖像輪廓
        base_blur = cv2.GaussianBlur(base, (3, 3), 0)
        _, base_thresh = cv2.threshold(base_blur, 127, 255, cv2.THRESH_BINARY_INV)
        contours_base, _ = cv2.findContours(base_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        for name, metric in metrics.items():
            scores = []
            for level in levels:
                distorted = apply_perturbation(base.copy(), level)

                if name == 'FRSS':
                    blur = cv2.GaussianBlur(distorted, (3, 3), 0)
                    _, threshed = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)
                    contours_dist, _ = cv2.findContours(threshed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

                    score = frss_shape_similarity(contours_base, contours_dist)
                else:
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

# ==== 4. 指標函數 ====
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
    folder = "benchmarkimg"
    font = "benchmarkimg/NotoSansTC-VariableFont_wght.ttf"

    metrics_dict = {
        'MSE': mse_similarity,
        'RMSE': rmse_similarity,
        'PSNR': psnr_similarity,
        'SSIM': ssim_similarity,
        'FRSS': frss_shape_similarity,  # ✅ 來自外部引用
    }

    evaluate_images_multiple_metrics(folder, metrics=metrics_dict, font_path=font)
