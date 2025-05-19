import cv2
import numpy as np
from scipy.ndimage import convolve

# --------- 功能函數們 ---------



# 模擬簡單 CNN 卷積：使用 Sobel 邊緣檢測或自定義 kernel
def simple_convolution(img, mode='sobel'):
    img = img.astype(np.float32) / 255.0

    if mode == 'sobel':
        kernel_x = np.array([[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]], dtype=np.float32)
        kernel_y = np.array([[1, 2, 1],
                             [0, 0, 0],
                             [-1, -2, -1]], dtype=np.float32)
        gx = convolve(img, kernel_x)
        gy = convolve(img, kernel_y)
        return np.sqrt(gx**2 + gy**2)  # magnitude
    elif mode == 'blur':
        kernel = np.ones((3, 3), dtype=np.float32) / 9
        return convolve(img, kernel)
    else:
        raise ValueError("Unsupported mode")

# 計算捲機後圖像差異（L1 和 L2 差）
def compare_convolved(conv1, conv2):
    l1 = np.mean(np.abs(conv1 - conv2))
    l2 = np.mean((conv1 - conv2)**2)
    return l1, l2

# --------- 主流程範例 ---------

# 載入灰階圖像
imgA = cv2.imread("target.png", cv2.IMREAD_GRAYSCALE)
imgB = cv2.imread("test2.png", cv2.IMREAD_GRAYSCALE)



# 步驟2：捲機（模擬 CNN 特徵萃取）
convA = simple_convolution(imgA, mode='sobel')
convB = simple_convolution(imgB, mode='sobel')

# 步驟3：比較卷積後特徵圖的像素差異
l1_diff, l2_diff = compare_convolved(convA, convB)

print(f"L1 Difference: {l1_diff:.4f}")
print(f"L2 Difference: {l2_diff:.4f}")
