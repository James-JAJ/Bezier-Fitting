# -*- coding: utf-8 -*-
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

from ...Bezier_Fitting.utils import *


# 主程式
if __name__ == "__main__":
    # --- 可調參數 ---
    image_path = 'A.png'
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
                debug=debug,
                ifserver=0
            )
            path = [fixcontour[i] for i in range(custom_idx[0], custom_idx[-1])]

                        # 取得該輪廓在 custom_idx 上切割的各段曲線
            result = []  # 所有段落的控制點
            final = vis_img.copy()  # 用於儲存繪圖結果
            width, height = vis_img.shape[1], vis_img.shape[0]
            predict = np.zeros_like(final)  # 預測圖層

            for i in range(len(custom_idx) - 1):
                start = custom_idx[i]
                end = custom_idx[i + 1]
                if end <= start:
                    continue  # 避免無效段

                # 擷取這段範圍內的點
                target_curve = path[start:end + 1]
                target_curve = [(int(p[0]), int(p[1])) for p in target_curve]

                # 除錯輸出
                custom_print(f"Line {i}: {target_curve[0]} -> {target_curve[-1]}")

                # 使用 genetic_algorithm 擬合控制點
                control_points = genetic_algorithm(
                    target_curve, 
                    target_curve[0], 
                    target_curve[-1], 
                    width, height,
                    ifserver=0
                )
                result.append(control_points)

                # 計算貝茲曲線座標並繪製到預測圖上
                curve_points = bezier_curve_calculate(control_points)
                predict = draw_curve_on_image(predict, curve_points, 3)

            # 最後將預測圖層疊加到原圖上顯示
            final = stack_image(final, predict)
            showimg("曲線擬合結果", final, 1)

            """
            print("自訂演算法簡化後的點數:", len(custom_points))
            
            # 繪製原始輪廓
            cv2.drawContours(vis_img, [contour], -1, (0, 255, 0), 1)
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