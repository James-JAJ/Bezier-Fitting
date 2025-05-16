# -*- coding: utf-8 -*-
import cv2
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')
from utils import *

# 主程式
if __name__ == "__main__":
    # --- 可調參數 ---
    image_path = 'C.png'
    scale_factor = 2             # 前處理放大倍數
    final_shrink_factor = 0.5    # 縮小倍數
    blur_ksize = 3               # 模糊核大小  
    threshold_value = 180        # 二質化閾值
    epsilon = 1.0                # 簡化輪廓的誤差
    rdp_epsilon = 3              # RDP簡化閾值
    curvature_threshold = 35     # 曲率閾值
    min_radius = 10              # 最小搜尋半徑
    max_radius = 50              # 最大搜尋半徑
    debug = True                 # 是否打印除錯信息
    ifshow = 0                   # 是否中途顯示
    # ----------------

    try:
        # 原圖 灰階圖
        original_img, gray_img = inputimg_colortogray(image_path)
        # 前處理圖片
        preprocessed_img = preprocess_image(gray_img, scale_factor, blur_ksize, threshold_value, ifshow)
        # 得到圖片輪廓
        contours = getContours(preprocessed_img, ifshow)
        # 縮小座標圖片
        vis_img = original_img.copy()
        contours = shrink_contours(contours, final_shrink_factor) 
        
        # 處理每個輪廓
        for contour in contours: 
            # 將輪廓轉換成點集
            fixcontour = [sublist[0] for sublist in contour]
            #print(fixcontour)

            # 使用RDP演算法簡化
            rdp_points = rdp(fixcontour, epsilon=rdp_epsilon)
            print("RDP簡化後的點數:", len(rdp_points))
            
            # 使用改進的自訂演算法進一步簡化
            custom_points, custom_idx = svcfp_queue(
                fixcontour, 
                rdp_points,
                min_radius=min_radius, 
                max_radius=max_radius, 
                curvature_threshold=curvature_threshold,
                debug=debug
            )
            print("自訂演算法簡化後的點數:", len(custom_points))
            
            # 繪製原始輪廓
            cv2.drawContours(vis_img, [contour], -1, (0, 255, 0), 1)
            
            """
            # 繪製RDP簡化後的點（紅色）
            for point in rdp_points:
                cv2.circle(vis_img, (point[0], point[1]), 3, (0, 0, 255), -1)
            
            # 繪製自訂演算法簡化後的點（藍色）
            for point in custom_points:
                cv2.circle(vis_img, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)
            """
            
    
        
        # 顯示結果
        showimg("輪廓簡化結果", vis_img, 1)
        
    except Exception as e:
        print(f"發生錯誤: {e}")
        import traceback
        traceback.print_exc()