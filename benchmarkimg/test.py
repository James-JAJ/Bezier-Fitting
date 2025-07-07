import cv2
import numpy as np
from scipy.spatial import KDTree
import sys 
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
def compare_svg_png_similarity(bezier_list, png_path, num_points=100):
    """
    比較貝茲曲線控制點集 vs 圖像輪廓的相似度，回傳 0~1 分數（越高越像）
    """
    

    # Step 1: 將 SVG 曲線轉為點雲
    svg_points = []
    for ctrl_pts in bezier_list:
        curve = bezier_curve_calculate(ctrl_pts, num_of_points=num_points)
        svg_points.extend(curve)
    svg_points = np.array(svg_points)

    # Step 2: 將 PNG 圖像轉為點雲
    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"圖片無法讀取：{png_path}")
    _, binary = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_points = np.vstack([c.squeeze() for c in contours if c.shape[0] >= 5])

    # Step 3: 雙向平均最近距離
    if len(svg_points) < 2 or len(img_points) < 2:
        return 0.0

    tree_svg = KDTree(svg_points)
    tree_img = KDTree(img_points)
    d1, _ = tree_svg.query(img_points)
    d2, _ = tree_img.query(svg_points)
    avg_d = (np.mean(d1) + np.mean(d2)) / 2

    # Step 4: 相似度指標（越小距離越高分）
    return float(1 / (1 + avg_d))
score = compare_svg_png_similarity(bezier_list, "test/B.png")
print(f"🧪 SFSS Score: {score:.4f}")
