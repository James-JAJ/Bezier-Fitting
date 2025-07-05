import cv2
import numpy as np
import sys
import os
import base64
from .server_tools import *
import svgwrite
from collections import defaultdict
from PIL import Image
from typing import List, Tuple, Optional
import math

# 類型別名
Point = Tuple[float, float]
BezierCurve = Tuple[Point, Point, Point, Point]  # (p0, p1, p2, p3)
# ================================================================
# ✅ 影像處理相關常用工具（預處理、輪廓、疊圖、儲存、SVG輸出等）
# ================================================================
def inputimg_colortobinary(imgpath):
    """
    ✅ 將彩色圖片轉為二進制黑白圖（128為閾值）

    Args:
        imgpath (str): 圖片檔案路徑（應為三通道圖片）

    Returns:
        np.ndarray (H, W): 二值化單通道圖片，pixel ∈ {0, 255}

    ⚠️ 備註：
        - 此函數直接使用 Python list 做初始轉換，再變成 numpy 陣列
        - 較慢，但有助於手動調整二值化規則（如需快速建議直接用 threshold）
    """
    img = cv2.imread(imgpath, 0)
    binary_img = [[0 if pixel < 128 else 255 for pixel in row] for row in img]
    binary_img = np.array(binary_img, dtype=np.uint8)
    return binary_img
def inputimg_colortogray(imgpath):
    """
    ✅ 將彩色圖轉為灰階並一併回傳原圖

    Args:
        imgpath (str): 圖片路徑

    Returns:
        tuple:
            img (np.ndarray): 原始彩色圖片（H, W, 3）
            img_gray (np.ndarray): 灰階單通道圖片（H, W）

    ⚠️ 若找不到圖會主動 raise FileNotFoundError
    """
    img = cv2.imread(imgpath)
    if img is None:
        raise FileNotFoundError(f"無法讀取圖像: {imgpath}")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    return img, img_gray
def showimg(img, name="test", ifshow=1):
    """
    ✅ 顯示圖片（用 OpenCV 彈窗方式）

    Args:
        img (np.ndarray): 要顯示的圖片
        name (str): 視窗名稱
        ifshow (int): 若為 1 則顯示，0 則跳過（方便關閉預覽）
    """
    if ifshow == 1:
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
def save_image(image, filename, path, ifserver):
    """
    ✅ 儲存圖像到指定資料夾，支援伺服器輸出回饋

    Args:
        image (np.ndarray): 要儲存的圖片
        filename (str): 儲存檔名
        path (str): 儲存資料夾路徑
        ifserver (int): 控制是否在伺服器輸出訊息
    """
    cv2.imwrite(path + "/" + filename, image)
    custom_print(ifserver, f"Image saved: {path}")
def encode_image_to_base64(image):
    """
    ✅ 將圖像轉為 base64 字串，方便 JSON 或 HTML 傳輸

    Args:
        image (np.ndarray): 圖片（灰階或彩色皆可）

    Returns:
        str: base64 編碼字串
    """
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')
def stack_image(image1, image2):
    """
    ✅ 將兩張圖片疊合，處理黑底遮罩

    Args:
        image1, image2 (np.ndarray): 要合併的兩張圖（建議相同大小）

    Returns:
        np.ndarray: 疊合後的新圖片
    """
    mask1 = cv2.inRange(image1, 0, 0)
    mask2 = cv2.inRange(image2, 0, 0)
    mask1_inv = cv2.bitwise_not(mask1)
    mask2_inv = cv2.bitwise_not(mask2)
    image1_fg = cv2.bitwise_and(image1, image1, mask=mask1_inv)
    image2_fg = cv2.bitwise_and(image2, image2, mask=mask2_inv)
    combined_image = cv2.add(image1_fg, image2_fg)
    return combined_image
def preprocess_image(img_gray, scale_factor=2, blur_ksize=3, threshold_value=200, ifshow=0):
    """
    ✅ 圖片預處理：放大 → 模糊 → 二值化

    Args:
        img_gray (np.ndarray): 灰階圖片
        scale_factor (int): 放大倍數
        blur_ksize (int): 模糊核大小（需為奇數）
        threshold_value (int): 二值化閾值
        ifshow (int): 是否顯示每步結果

    Returns:
        np.ndarray: 處理後二值圖（黑白）

    💡 常用於準備輪廓提取（如 cv2.findContours）
    """
    height, width = img_gray.shape
    resized = cv2.resize(img_gray, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_CUBIC)
    showimg(resized, "resized", ifshow)
    blurred = cv2.GaussianBlur(resized, (blur_ksize, blur_ksize), 0)
    showimg(blurred, "blurred", ifshow)
    _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)
    showimg(binary, "binary", ifshow)
    return binary
def getContours(binary_img, ifshow=0):
    """
    ✅ 取得輪廓（含可視化）

    Args:
        binary_img (np.ndarray): 二值圖（黑底白字）
        ifshow (int): 是否顯示輪廓圖

    Returns:
        list[np.ndarray]: OpenCV 輪廓格式（每條 contour 為點陣列）
    """
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    vis_img = cv2.cvtColor(binary_img.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 1)
    showimg(vis_img, "contours", ifshow)
    return contours
def generate_closed_bezier_svg(bezier_ctrl_points, width, height, filename="closed_bezier.svg"):
    """
    ✅ 生成不填色的 SVG（使用 M...C 貝茲格式）

    Args:
        bezier_ctrl_points (list[list[tuple]]): 所有貝茲控制點
        width, height (int): SVG 畫布大小
        filename (str): 輸出檔名

    備註：
        - 每段貝茲需為四點格式（P0, P1, P2, P3）
        - 不會閉合，只畫線條，適合進一步嵌套處理
    """
    dwg = svgwrite.Drawing(filename, profile='full', size=(width, height))
    for ctrl_pts in bezier_ctrl_points:
        P0, P1, P2, P3 = ctrl_pts
        d = f"M {P0[0]} {P0[1]} C {P1[0]} {P1[1]}, {P2[0]} {P2[1]}, {P3[0]} {P3[1]}"
        path = dwg.path(d=d, fill='none', stroke='black', stroke_width=1)
        dwg.add(path)
    dwg.save()
    print(f"✅ SVG 輸出完成: {filename}")
def get_contour_levels(hierarchy):
    """
    ✅ 根據 hierarchy 決定每個輪廓的嵌套層級

    Args:
        hierarchy (np.ndarray): OpenCV findContours 的 hierarchy 結果（通常為 (1, N, 4)）

    Returns:
        list[int]: 每個輪廓對應的層級（0 為最外層）

    備註：
        - parent = h[3]；向上回推可得當前深度
        - 可用於 SVG 填色規則（偶數白，奇數空白）
    """
    levels = []
    for h in hierarchy:
        level = 0
        parent = h[3]
        while parent != -1:
            level += 1
            parent = hierarchy[parent][3]
        levels.append(level)
    return levels
def fill_small_contours(img, area_threshold=3000):
    """
    填充小輪廓
    """
    # 若為灰階就不再轉換
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    result = img.copy()
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < area_threshold:
            cv2.drawContours(result, [contour], -1, (0, 0, 255), thickness=cv2.FILLED)

    if len(img.shape) == 3 and img.shape[2] == 3:
        line_mask = cv2.inRange(img, (0, 0, 100), (100, 100, 255))
        result[line_mask > 0] = [0, 0, 255]

    return result





