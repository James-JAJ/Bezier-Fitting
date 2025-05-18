import cv2
import numpy as np
import sys
import os
import base64
from .server_tools import *
def inputimgcolortobinary(imgpath):
    """
    輸入圖片回傳二進制檔
    Args:
        imgpath (str): 三通道彩色圖片路徑
    Returns:
        img    (list): 二進制單通道圖片
    Waring:
        img 以128為閾值進行以128為閾值進行二元化
    """
    img = cv2.imread(imgpath, 0)  # 讀取圖片為灰階
    # 將圖像二元化
    binary_img = [[0 if pixel < 128 else 255 for pixel in row] for row in img]
    # 將二元化圖像轉換為numpy數組
    binary_img = np.array(binary_img, dtype=np.uint8)
    return img
def inputimg_colortogray(imgpath):
    """
    輸入圖片回傳灰階圖片
    Args:
        orgimg       (str): 三通道彩色圖片路徑路徑
    Returns:
        img_gray    (list): 灰階單通道圖片
        img         (list): 原圖
    """
    img = cv2.imread(imgpath)
    if img is None:
        raise FileNotFoundError(f"無法讀取圖像: {imgpath}")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    return img, img_gray
def showimg(img, name="test", ifshow=1):
    """
    在本地端顯示圖片
    Args:
        img     (list): 圖片
        ifshow   (int): 是否顯示圖片
    """
    if ifshow==1:
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
def save_image(image, filename,path,ifserver):
    """
    儲存圖片(未修改路徑)
    Args:
        image      (list): 圖片
        filename    (str): 檔名
    """
    cv2.imwrite(path+"/"+filename, image)
    custom_print(ifserver,f"Image saved: {path}")
def encode_image_to_base64(image):
    """
    encode_image_to_base64:
    Args:
        image      (list): 圖片
    Returns:
        list: 轉換後圖片
    """
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')
def stack_image(image1, image2):
    """
    將兩張圖片疊合
    Args:
        image1, image2  (list): 雙圖片疊圖
    Returns:
        combined_image  (list): 疊圖後圖片
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
    輸入圖片回傳灰階圖片
    Args:
        img_gray           (list): 灰階圖片大小
        scale_factor        (int): 圖片縮放倍率
        blur_ksize          (int): 模糊核大小(必須大於1的奇數)
        threshold_value     (int): 二值化閾值
        ifshow             (bool): 是否顯示圖片
    Returns:
        binary             (list): 灰階單通道圖片
    """
    height, width = img_gray.shape
    resized = cv2.resize(img_gray, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_CUBIC)
    showimg( resized,"resized", ifshow)

    blurred = cv2.GaussianBlur(resized, (blur_ksize, blur_ksize), 0)
    showimg( blurred,"blurred", ifshow)

    _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)
    showimg( binary, "binary",ifshow)

    return binary
def getContours(binary_img, ifshow=0):
    """
    取得灰階圖片並取得輪廓
    Args:
        binary_img         (list): 灰階圖片
        ifshow             (bool): 是否顯示圖片
    """
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    vis_img = cv2.cvtColor(binary_img.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 1)
    showimg(vis_img,"contours", ifshow)
    return contours