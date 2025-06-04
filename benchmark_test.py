import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib import font_manager
from typing import Callable, Dict
from skimage.metrics import structural_similarity as ssim
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *  # 導入所有工具函數，包括 server_tools 中的函數
sys.stdout.reconfigure(encoding='utf-8')  # 改變輸出的編碼

# Red 輪廓圖直接取黑色線條點雲
def image_to_points_from_lines(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((1, 2))
    points = np.column_stack(np.where(img < 127))
    return points[:, [1, 0]] if len(points) > 0 else np.zeros((1, 2))


# ==== 擾動函數（旋轉 + 縮放 + 平移 + 模糊） ====
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

# ==== 主程式 ====
def evaluate_images_multiple_metrics(
    folder_path: str,
    metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]],
    levels: np.ndarray = np.linspace(0, 1, 10),
    font_path: str = None
):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg'))]
    results = {name: [] for name in metrics}

    for file in image_files:
        print(f"處理圖像: {file}")
        path = os.path.join(folder_path, file)
        base = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        base = cv2.resize(base, (256, 256))

        base_points = image_to_points_from_lines(path)

        for name, metric in metrics.items():
            scores = []
            for level in levels:
                try:
                    distorted = apply_perturbation(base.copy(), level)
                    distorted_path = os.path.join(folder_path, f"_temp_{file}_lv{level:.2f}.png")
                    cv2.imwrite(distorted_path, distorted)
                    distorted_points = image_to_points_from_lines(distorted_path)

                    if name == 'SCS':
                        score = scs_shape_similarity(base_points, distorted_points)
                    else:
                        score = metric(base, distorted)

                    scores.append(score)
                    os.remove(distorted_path)
                except Exception as e:
                    print(f"  警告: {name} 在 level {level:.2f} 時出錯: {e}")
                    scores.append(0.0)
            results[name].append(scores)
            print(f"  {name}: 平均分數 = {np.mean(scores):.3f}")

    # 畫圖
    plt.figure(figsize=(12, 6), dpi=100)
    for name, score_matrix in results.items():
        score_array = np.array(score_matrix)
        means = np.mean(score_array, axis=0)
        stds = np.std(score_array, axis=0)
        plt.errorbar(levels, means, yerr=stds, fmt='o-', capsize=3, label=name, alpha=0.8)

    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.xlabel("擾動程度", fontproperties=font_manager.FontProperties(fname=font_path) if font_path else None)
    plt.ylabel("相似度", fontproperties=font_manager.FontProperties(fname=font_path) if font_path else None)
    plt.title("多指標相似度比較 (修正版)", fontproperties=font_manager.FontProperties(fname=font_path) if font_path else None)
    plt.legend(prop=font_manager.FontProperties(fname=font_path) if font_path else None)
    plt.tight_layout()
    plt.show()

# ==== 類型為影像比較的其他指標 ====
def ssim_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    try:
        return ssim(img1, img2)
    except:
        return 0.0

def mse_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    try:
        err = np.mean((img1.astype("float") - img2.astype("float")) ** 2)
        return 1 / (1 + err)
    except:
        return 0.0

def rmse_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    try:
        mse = np.mean((img1.astype("float") - img2.astype("float")) ** 2)
        rmse = np.sqrt(mse)
        return 1 / (1 + rmse)
    except:
        return 0.0

def psnr_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    try:
        mse = np.mean((img1.astype("float") - img2.astype("float")) ** 2)
        if mse == 0:
            return 1.0
        PIXEL_MAX = 255.0
        psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
        return min(1.0, psnr / 100)
    except:
        return 0.0

# ==== 執行 ====
if __name__ == '__main__':
    folder = "benchmarkimg"
    font = "benchmarkimg/NotoSansTC-VariableFont_wght.ttf"
    metrics_dict = {
        'MSE': mse_similarity,
        'RMSE': rmse_similarity,
        'PSNR': psnr_similarity,
        'SSIM': ssim_similarity,
        'SCS': lambda a, b: 0.0  # dummy，會被替換
    }
    # 將主程式中直接改為呼叫 scs_from_pointcloud
    evaluate_images_multiple_metrics(folder, metrics=metrics_dict, font_path=font)
