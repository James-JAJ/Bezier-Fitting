import cv2
import numpy as np
import sys
import os
import base64
from .server_tools import *
def inputimg_colortobinary(imgpath):
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


def fill_contours_only(binary_img):
    """
    偵測圖片白色區域並根據層次關係填充：
    1. 偵測白色區域的層次關係
    2. 基數層次(1,3,5...)填黑色，偶數層次(0,2,4...)填白色
    3. 輸出白底黑線圖
    4. 保留原本非白色的線條
    
    Args:
        input_img: 輸入圖像
    Returns:
        result_img: 處理後的白底黑線圖像
    """
    # 轉換為灰階
    if len(binary_img.shape) == 3:
        gray = cv2.cvtColor(binary_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = binary_img.copy()
    
    # 二值化處理，分離白色區域和非白色區域
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)  # 白色區域為255
    
    # 保存原始的非白色線條（黑色和其他顏色的線條）
    original_lines = (gray < 240).astype(np.uint8) * 255  # 非白色區域變為白色標記
    
    # 對白色區域進行輪廓檢測
    # 需要反轉來找白色區域的邊界
    white_areas = binary.copy()
    
    # 形態學處理，清理白色區域
    kernel = np.ones((3,3), np.uint8)
    white_areas = cv2.morphologyEx(white_areas, cv2.MORPH_CLOSE, kernel)
    white_areas = cv2.morphologyEx(white_areas, cv2.MORPH_OPEN, kernel)
    
    # 反轉圖像來檢測白色區域的輪廓
    inverted = 255 - white_areas
    
    # 找輪廓，獲取層次結構
    contours, hierarchy = cv2.findContours(inverted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 創建結果圖像，初始為白色背景
    result = np.ones_like(gray) * 255
    
    if hierarchy is not None and len(contours) > 0:
        hierarchy = hierarchy[0]  # 展開hierarchy維度
        
        # 計算每個輪廓的嵌套深度
        def calculate_nesting_depth(contour_idx):
            """計算輪廓的嵌套深度，最外層為0"""
            depth = 0
            parent_idx = hierarchy[contour_idx][3]  # 父輪廓索引
            while parent_idx != -1:
                depth += 1
                parent_idx = hierarchy[parent_idx][3]
            return depth
        
        # 為每個輪廓計算深度和面積
        contour_info = []
        for i in range(len(contours)):
            depth = calculate_nesting_depth(i)
            area = cv2.contourArea(contours[i])
            contour_info.append((i, depth, area))
            print(f"白色區域輪廓 {i}: 深度={depth}, 面積={area:.1f}")
        
        # 按深度分組
        depth_groups = {}
        for contour_idx, depth, area in contour_info:
            if depth not in depth_groups:
                depth_groups[depth] = []
            depth_groups[depth].append((contour_idx, area))
        
        print(f"檢測到的白色區域深度層級: {sorted(depth_groups.keys())}")
        
        # 按深度從深到淺處理
        for depth in sorted(depth_groups.keys(), reverse=True):
            contours_at_depth = depth_groups[depth]
            
            # 基數層次(1,3,5...)填黑色，偶數層次(0,2,4...)保持白色
            if depth % 2 == 1:  # 基數層次
                for contour_idx, area in contours_at_depth:
                    if area > 10:  # 過濾太小的區域
                        cv2.fillPoly(result, [contours[contour_idx]], 0)  # 填黑色
                        print(f"填充深度 {depth} 的白色區域 {contour_idx} (面積: {area:.1f}) - 黑色")
            else:  # 偶數層次
                for contour_idx, area in contours_at_depth:
                    if area > 10:
                        cv2.fillPoly(result, [contours[contour_idx]], 255)  # 保持白色
                        print(f"保持深度 {depth} 的白色區域 {contour_idx} (面積: {area:.1f}) - 白色")
    
    # 將原始的非白色線條疊加到結果上（變成黑色線條）
    # 找出原始圖像中非白色的像素位置
    non_white_mask = (gray < 240)  # 非白色區域的遮罩
    result[non_white_mask] = 0  # 將非白色區域在結果圖中設為黑色
    
    print("處理完成：基數層次填黑色，偶數層次保持白色，原始線條保留為黑色")
    
    return result






