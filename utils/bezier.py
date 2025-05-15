import cv2
import numpy as np
#bezier check
#bezier_curve_calculate:使用四點計算貝茲曲線
def bezier_curve_calculate(points, num_of_points=50):
    """生成貝茲曲線上的點，強制轉換為 Python 標量"""
    curve_points = []
    for t in np.linspace(0, 1, num_of_points):
        x = float((1 - t)**3 * points[0][0] + 3 * (1 - t)**2 * t * points[1][0] + 3 * (1 - t) * t**2 * points[2][0] + t**3 * points[3][0])
        y = float((1 - t)**3 * points[0][1] + 3 * (1 - t)**2 * t * points[1][1] + 3 * (1 - t) * t**2 * points[2][1] + t**3 * points[3][1])
        curve_points.append((int(x), int(y)))
    return curve_points
#draw_curve_on_image:在圖片上畫出貝茲曲線
def draw_curve_on_image(img,curve_points,A):
    for m in range(len(curve_points) - 1):
        curve = cv2.line(img, curve_points[m], curve_points[m+1], (255, 255, 255), A)
    return img
