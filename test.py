import cv2
import numpy as np
from scipy.spatial import cKDTree
import math
# -*- coding: utf-8 -*-
import sys
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory, Response
import time
import threading
import os
from utils import *  # 導入所有工具函數，包括 server_tools 中的函數
#print(os.getcwd())
#system initialization
sys.stdout.reconfigure(encoding='utf-8')  # 改變輸出的

# 重新載入圖片（因 code 執行環境重置）

img_path1 = "img/152115_80.71715_27_(0.01s)_inkscape.png"
img_path2 = "img/152115_80.71715_27_(0.01s)_orig.png"
"""
img_path1 = "test/NO_CNN_start.(147, 46)end.(485, 423).fitted.png"
img_path2 = "test/NO_CNN_start.(147, 46)end.(485, 423).Original.png"
img_path1 = "img/0.528_14_orig.png"
img_path2 = "img/0.528_14_fitting.png"
img_path1 = "benchmarkimg/B.png"
img_path2 = "benchmarkimg/red_layer.png"
"""


img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)

# 二值化 + 輪廓提取
_, bin1 = cv2.threshold(img1, 200, 255, cv2.THRESH_BINARY)
_, bin2 = cv2.threshold(img2, 200, 255, cv2.THRESH_BINARY)
showimg(img1)
showimg(img2)
A = np.argwhere(bin1==0)
B = np.argwhere(bin2==0)
print(A)


sim = scs_shape_similarity(A, B)


print(sim)
