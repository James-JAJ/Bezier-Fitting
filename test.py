import sys
import os
import cv2
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
image_pathA="benchmarkimg/B.png"
image_pathB="benchmarkimg/B.png"

original_imgA, gray_imgA = inputimg_colortogray(image_pathA)
original_imgB, gray_imgB = inputimg_colortogray(image_pathB)

# ✅ 正確的前處理
gray_imgA = cv2.GaussianBlur(gray_imgA, (3, 3), 0)
_, gray_imgA = cv2.threshold(gray_imgA, 200, 255, cv2.THRESH_BINARY_INV)

gray_imgB = cv2.GaussianBlur(gray_imgB, (3, 3), 0)
_, gray_imgB = cv2.threshold(gray_imgB, 200, 255, cv2.THRESH_BINARY_INV)

A, _ = cv2.findContours(gray_imgA, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
B, _ = cv2.findContours(gray_imgB, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

value = frss_shape_similarity(A, B)
print(value)
