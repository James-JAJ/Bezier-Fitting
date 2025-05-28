# -*- coding: utf-8 -*-
import numpy as np
import sys
import os
import cv2
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *

# 主程式
if __name__ == "__main__":
    # --- 可調參數 ---
    image_path = 'test/B.png'
    scale_factor = 2             # 前處理放大倍數
    final_shrink_factor = 0.5    # 縮小倍數
    blur_ksize = 3               # 模糊核大小  
    threshold_value = 200        # 二質化閾值
    epsilon = 1.0                # 簡化輪廓的誤差
    rdp_epsilon = 2              # RDP簡化閾值
    curvature_threshold = 42     # 曲率閾值
    min_radius = 10              # 最小搜尋半徑
    max_radius = 50              # 最大搜尋半徑
    insert_threshold = 100
    fuse_radio = 5
    fuse_threshold = 10
    debug = True                 # 是否打印除錯信息
    ifshow = 0                   # 是否中途顯示
    # ----------------

    try:
        # 原圖 灰階圖
        original_img, gray_img = inputimg_colortogray(image_path)
        preprocessed_img = preprocess_image(gray_img, scale_factor, blur_ksize, threshold_value, ifshow)
        contours = getContours(preprocessed_img, ifshow)

        AAA = original_img.copy()
        contours = shrink_contours(contours, final_shrink_factor) 
        cv2.drawContours(AAA, contours, -1, (0, 255, 0), 1)
        #showimg(AAA)

        vis_img  = original_img.copy()
        red_layer = np.ones_like(vis_img) * 255
        predict = np.zeros_like(vis_img.copy())
        pointtotal = 0
        rdptotal = 0

        for contour in contours:
            fixcontour = [sublist[0] for sublist in contour]
            fixcontour = remove_consecutive_duplicates(fixcontour)
            rdp_points = rdp(fixcontour, epsilon=rdp_epsilon)
            print("RDP簡化後的點數:", len(rdp_points))
            rdptotal += len(rdp_points)

            custom_points, custom_idx = svcfp(
                fixcontour,
                min_radius=min_radius,
                max_radius=max_radius,
                curvature_threshold=curvature_threshold,
                rdp_epsilon=rdp_epsilon,
                insert_threshold=insert_threshold,
                fuse_radio=fuse_radio,
                fuse_threshold=fuse_threshold,
                ifserver=0
            )

            pointtotal += len(custom_points)
            path = fixcontour
            cv2.drawContours(vis_img, [contour], -1, (0, 255, 0), 1)

            for i in range(len(custom_idx)):
                print(path[custom_idx[i]])

            for i in range(len(custom_idx) - 1):
                start = custom_idx[i]
                end = custom_idx[i + 1]
                target_curve = path[start:end]
                target_curve = np.array([(int(p[0]), int(p[1])) for p in target_curve])
                custom_print(0, f"Line {i}: {target_curve[0]} -> {target_curve[-1]}")

                ctrl_pts = fit_least_squares_bezier(target_curve)
                curve_points = bezier_curve_calculate(ctrl_pts)
                vis_img = draw_curve_on_image(vis_img, curve_points, 1)
                red_layer = draw_curve_on_image(red_layer, curve_points, 1, (0, 0, 255))

        print("Total points:", pointtotal)
        print("Total RDP points:", rdptotal)

        showimg(vis_img, "輪廓簡化結果", 1)
        final = stack_image(vis_img.copy(), predict)
        showimg(final, "疊圖結果", 1)
        # 顯示純紅線圖層
        red_layer = fill_contours_only(red_layer)

        showimg(red_layer, "紅線擬合圖", 1)

    except Exception as e:
        print(f"發生錯誤: {e}")
        import traceback
        traceback.print_exc()
