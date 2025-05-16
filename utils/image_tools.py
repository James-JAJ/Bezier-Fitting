import cv2
import numpy as np
import sys
import os
import base64
from .server_tools import *
#img process
#inputimg:輸入圖片回傳二進制檔
def inputimgcolortobinary(path):
    img = cv2.imread(path, 0)  # 讀取圖片為灰階
    # 將圖像二元化
    binary_img = [[0 if pixel < 128 else 255 for pixel in row] for row in img]
    # 將二元化圖像轉換為numpy數組
    binary_img = np.array(binary_img, dtype=np.uint8)
    return img
def inputimg_colortogray(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"無法讀取圖像: {image_path}")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    return img, img_gray
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
def preprocess_image(img_gray, scale_factor=2, blur_ksize=3, threshold_value=200, ifshow=0):
    height, width = img_gray.shape
    resized = cv2.resize(img_gray, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_CUBIC)
    showimg("resized", resized, ifshow)

    blurred = cv2.GaussianBlur(resized, (blur_ksize, blur_ksize), 0)
    showimg("blurred", blurred, ifshow)

    _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)
    showimg("binary", binary, ifshow)

    return binary

def getContours(binary_img, ifshow=0):
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    vis_img = cv2.cvtColor(binary_img.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 1)
    showimg("contours", vis_img, ifshow)