import cv2
import numpy as np
import sys
import os
import base64
from .server_tools import *
#img process
#inputimg:輸入圖片回傳二進制檔
def inputimg(path):
    img = cv2.imread(path, 0)  # 讀取圖片為灰階
    # 將圖像二元化
    binary_img = [[0 if pixel < 128 else 255 for pixel in row] for row in img]
    # 將二元化圖像轉換為numpy數組
    binary_img = np.array(binary_img, dtype=np.uint8)
    return img
#showimg:在本地端顯示圖片
def showimg(img):
    cv2.imshow('Bezier Curve', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#save_image:儲存圖片
def save_image(image, filename,path="AAA/"):
    cv2.imwrite(path, image)
    custom_print(f"Image saved: {path}")
#encode_image_to_base64:將圖片轉換為base64編碼
def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')
#stack_image:將兩張圖片疊合
def stack_image(image1, image2):
    mask1 = cv2.inRange(image1, 0, 0)
    mask2 = cv2.inRange(image2, 0, 0)
    mask1_inv = cv2.bitwise_not(mask1)
    mask2_inv = cv2.bitwise_not(mask2)
    image1_fg = cv2.bitwise_and(image1, image1, mask=mask1_inv)
    image2_fg = cv2.bitwise_and(image2, image2, mask=mask2_inv)
    combined_image = cv2.add(image1_fg, image2_fg)
    return combined_image