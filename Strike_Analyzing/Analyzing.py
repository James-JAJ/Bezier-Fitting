import cv2
import numpy as np
import mahotas
from skimage.feature import hog
from skimage.transform import resize
import sys
sys.stdout.reconfigure(encoding='utf-8')

# --- 載入並處理圖片 ---
def load_image(path, size=(128, 128)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    img = cv2.resize(img, size)
    return img

# --- 1. Hu Moments ---
def hu_moments(img):
    moments = cv2.moments(img)
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)  # log-scale 處理
    return hu

# --- 2. Zernike Moments (使用 Mahotas) ---
def zernike_moments_mahotas(img, radius=64, degree=8):
    img = resize(img, (radius*2, radius*2))  # 轉成 Mahotas 要的大小
    img = (img > 0.5).astype(np.uint8)       # Mahotas 要求二值圖（0,1）
    return np.abs(mahotas.features.zernike_moments(img, radius, degree))


# --- 3. HOG 特徵 ---(X)
def hog_feature(img):
    return hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')

# --- 4. EMD ---(X)
def emd_hist(img1, img2, bins=16):
    h1 = cv2.calcHist([img1], [0], None, [bins], [0, 256])
    h2 = cv2.calcHist([img2], [0], None, [bins], [0, 256])

    # Normalize histograms
    h1 = cv2.normalize(h1, h1).flatten().astype(np.float32)
    h2 = cv2.normalize(h2, h2).flatten().astype(np.float32)

    # Convert to signature form: [value, bin_index]
    sig1 = np.array([[v, float(i)] for i, v in enumerate(h1)], dtype=np.float32)
    sig2 = np.array([[v, float(i)] for i, v in enumerate(h2)], dtype=np.float32)

    # Use OpenCV EMD
    emd_value, _, _ = cv2.EMD(sig1, sig2, cv2.DIST_L2)
    return emd_value


# --- 計算 L2 距離 ---
def l2_dist(v1, v2):
    return np.linalg.norm(v1 - v2)

# --- 全方法比較 ---
def compare_all_methods(imgA, imgB, label):
    print(f"\n📊 比對目標圖與 {label}：")

    huA, huB = hu_moments(imgA), hu_moments(imgB)
    print(f"Hu Moments 差異：\t\t{l2_dist(huA, huB):.4f}")

    zmA, zmB = zernike_moments_mahotas(imgA), zernike_moments_mahotas(imgB)
    print(f"Zernike Moments 差異：\t{l2_dist(zmA, zmB):.4f}")

    hogA, hogB = hog_feature(imgA), hog_feature(imgB)
    print(f"HOG 特徵差異：\t\t{l2_dist(hogA, hogB):.4f}")

    emd = emd_hist(imgA, imgB)
    print(f"EMD 差異：\t\t\t{emd:.4f}")

# --- 主程式 ---
if __name__ == "__main__":
    target = load_image("target.png")
    test1 = load_image("test1.png")
    test2 = load_image("test2.png")

    compare_all_methods(target, test1, "Test 1")
    compare_all_methods(target, test2, "Test 2")
