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
def draw_curve_on_image(image, curve_points, thickness=1, color=(0, 0, 255)):
    if len(curve_points) >= 2:
        pts = np.array(curve_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=False, color=color, thickness=thickness)
    return image
