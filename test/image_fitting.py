# -*- coding: utf-8 -*-
import numpy as np
import sys
import os
import cv2
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#print(os.getcwd())
from utils import *



# 主程式
if __name__ == "__main__":
    # --- 可調參數 ---
    image_path = 'test/D.png'
    scale_factor = 2             # 前處理放大倍數
    final_shrink_factor = 0.5    # 縮小倍數
    blur_ksize = 3               # 模糊核大小  
    threshold_value = 200        # 二質化閾值
    epsilon = 1.0                # 簡化輪廓的誤差
    rdp_epsilon = 2             # RDP簡化閾值
    curvature_threshold = 23    # 曲率閾值
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
        AAA = original_img.copy()
        contours = shrink_contours(contours, final_shrink_factor) 
        cv2.drawContours(AAA, contours, -1, (0, 255, 0), 1)
        vis_img  = original_img.copy()
        showimg(AAA)
        predict = np.zeros_like(vis_img.copy())  # 每次都使用同一張預測圖層來疊畫所有曲線
        pointtotal=0
        rdptotal=0
        # 處理每個輪廓
        for contour in contours:
            if len(contour)<=20:
                continue
            fixcontour = [sublist[0] for sublist in contour]
            fixcontour = remove_consecutive_duplicates(fixcontour)  # 移除首尾或相鄰重複點
            rdp_points = rdp(fixcontour, epsilon=rdp_epsilon)
            print("RDP簡化後的點數:", len(rdp_points))
            rdptotal+=len(rdp_points)
            custom_points, custom_idx = svcfp_queue(
                fixcontour,
                rdp_points,
                min_radius=min_radius,
                max_radius=max_radius,
                curvature_threshold=curvature_threshold,
                debug=debug,
                ifserver=0
            )
            pointtotal+=len(custom_points)
            path = fixcontour  # 用整個原始點序列來切

            width, height = vis_img.shape[1], vis_img.shape[0]
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
            #print(custom_points)
            for i in range(len(custom_idx)):
                print(path[custom_idx[i]])
            #
            for i in range(len(custom_idx) - 1):
                start = custom_idx[i]
                end = custom_idx[i + 1]
                target_curve = path[start:end]
                target_curve = np.array([(int(p[0]), int(p[1])) for p in target_curve])
                custom_print(0, f"Line {i}: {target_curve[0]} -> {target_curve[-1]}")
                #print(target_curve)
                if len(target_curve)<=10:
                    continue
                ctrl_pts = fit_fixed_end_bezier(target_curve, path[start],path[end])

                # 🎯 畫貝茲曲線在 vis_img 上（紅線）
                curve_points = bezier_curve_calculate(ctrl_pts)
                vis_img = draw_curve_on_image(vis_img, curve_points, 1)
        
        print(pointtotal)
        print(rdptotal)
        #GA
        """
        # 🎯 改為直接在原圖上畫貝茲線與節點
        for i in range(len(custom_idx) - 1):
            start = custom_idx[i]
            end = custom_idx[i + 1]
            print(start,end)
            target_curve = path[start:end]
            target_curve = np.array([(int(p[0]), int(p[1])) for p in target_curve])
            custom_print(0, f"Line {i}: {target_curve[0]} -> {target_curve[-1]}")
            #print(target_curve)
            ctrl_pts, max_error, mean_error = fit_and_evaluate_bezier(target_curve)

            # 🎯 畫貝茲曲線在 vis_img 上（紅線）
            curve_points = bezier_curve_calculate(ctrl_pts)
            vis_img = draw_curve_on_image(vis_img, curve_points, 2)
        """


        # 🎯 所有曲線畫完，再疊加到原圖上
        final = stack_image(vis_img.copy(), predict)
        showimg(final,"輪廓簡化結果", 1)
        
        
    except Exception as e:
        print(f"發生錯誤: {e}")
        import traceback
        traceback.print_exc()