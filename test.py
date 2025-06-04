import sys
import os
import cv2
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
image_pathA="benchmarkimg/NO_CNN_start.(101, 129)end.(357, 156).Original.png"
image_pathB="benchmarkimg/NO_CNN_start.(101, 129)end.(357, 156).RSME.png"
import cv2
import numpy as np
from scipy.spatial import cKDTree


def scs_shape_similarity(contours1, contours2):
    """
    SCS：Symmetric Contour Similarity
    接收兩組輪廓列表，回傳相似度值（0~1）
    """

    def contours_to_points(contours):
        if not contours:
            return np.zeros((1, 2))
        return np.concatenate([c.reshape(-1, 2) for c in contours if c.shape[0] >= 5])

    def mean_min_distance(A, B):
        tree = cKDTree(A)
        dists, _ = tree.query(B)
        return np.mean(dists)

    def symmetric_similarity(A, B):
        if len(A) < 2 or len(B) < 2:
            return 0.0
        avg_dist = (mean_min_distance(A, B) + mean_min_distance(B, A)) / 2
        return 1 / (1 + avg_dist)

    points1 = contours_to_points(contours1)
    points2 = contours_to_points(contours2)

    return float(symmetric_similarity(points1, points2))
imgA = cv2.imread("test/red_layer.png", 0)
imgB = cv2.imread("test/B.png", 0)
_, binA = cv2.threshold(imgA, 128, 255, cv2.THRESH_BINARY_INV)
_, binB = cv2.threshold(imgB, 128, 255, cv2.THRESH_BINARY_INV)

contoursA, _ = cv2.findContours(binA, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contoursB, _ = cv2.findContours(binB, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

score = scs_shape_similarity(contoursA, contoursB)
print("SCS 相似度分數：", score)
