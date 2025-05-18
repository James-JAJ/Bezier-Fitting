import cv2
import numpy as np
#bezier check
def bezier_curve_calculate(points, num_of_points=50):
    """
    由一組三次貝茲點生成貝茲曲線上的點，並強制轉換至 Python 純量
    Args:
        points          (list of tuple): 三次貝茲曲線之四點 Datatye: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    Returns:
        curve_points    (list of tuple): 由貝茲方程式所轉換的離散點數集陣列 Datatye: [(x1,y1), (x2,y2), ... (xn,yn)]
    Warning:
        points 點數集內部座標必須為整數
    """
    curve_points = []
    for t in np.linspace(0, 1, num_of_points):
        x = float((1 - t)**3 * points[0][0] + 3 * (1 - t)**2 * t * points[1][0] + 3 * (1 - t) * t**2 * points[2][0] + t**3 * points[3][0])
        y = float((1 - t)**3 * points[0][1] + 3 * (1 - t)**2 * t * points[1][1] + 3 * (1 - t) * t**2 * points[2][1] + t**3 * points[3][1])
        curve_points.append((int(x), int(y)))
    return curve_points
def draw_curve_on_image(img,curve_points,width):
    """
    在圖片上畫出貝茲曲線
    Args:
        img             (list of tuple): 三通道彩色圖片
        curve_points    (list of tuple): 經bezier_curve_calculate輸出曲線點位集 Datatye: [(x1,y1), (x2,y2), ... (xn,yn)]
        width           (int)          : 線段寬度線段寬度
    Returns:
        img             (list of tuple): 由貝茲方程式所轉換的離散點數集陣列 Datatye: [(x1,y1), (x2,y2), ... (xn,yn)]
    Waring:
        curve_points 點數集內部座標必須為整數
    """
    for m in range(len(curve_points) - 1):
        curve = cv2.line(img, curve_points[m], curve_points[m+1], (0, 255, 0), width)
    return img
