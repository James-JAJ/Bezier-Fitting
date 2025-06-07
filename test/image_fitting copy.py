# -*- coding: utf-8 -*- 
import numpy as np 
import sys 
import os 
import cv2 
import time 
from datetime import datetime 

sys.stdout.reconfigure(encoding='utf-8') 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

from utils import * 

image_dir = 'output_img' 
os.makedirs(image_dir, exist_ok=True) 

# 所有圖片 
image_files = [f for f in os.listdir('IMAGE') if f.lower().endswith(('.png', '.jpg', '.jpeg'))] 

# 擬合參數 
scale_factor = 2 
final_shrink_factor = 0.5 
blur_ksize = 3 
threshold_value = 200 
rdp_epsilon = 4 
curvature_threshold = 41 
min_radius = 10 
max_radius = 50 
insert_threshold = 100 
fuse_radio = 5 
fuse_threshold = 10 

for image_name in image_files: 
    try: 
        image_path = os.path.join('IMAGE', image_name) 
        orig_color, gray_img = inputimg_colortogray(image_path) 
        preprocessed = preprocess_image(gray_img, scale_factor, blur_ksize, threshold_value) 
        contours = getContours(preprocessed) 
        contours = shrink_contours(contours, final_shrink_factor) 

        height, width = gray_img.shape[:2] 
        orig = np.ones((height, width), dtype=np.uint8) * 255 
        for contour in contours: 

            for point in interpolate_points([pt[0] for pt in contour]): 
                x, y = map(int, point) 
                if 0 <= x < width and 0 <= y < height: 
                    orig[y][x] = 0

        final_ga = np.ones((height, width, 3), dtype=np.uint8) * 255 
        final_lstsq = np.ones((height, width, 3), dtype=np.uint8) * 255 
        only_ga = np.ones((height, width), dtype=np.uint8) * 255
        only_lstsq = np.ones((height, width), dtype=np.uint8) * 255

        all_custom_points = [] 
        pointtotal = 0 
        rdptotal = 0 

        # SVCFP + 最小二乘法 (完整流程时间测量)
        t1_svcfp_lstsq = time.time()
        for contour in contours: 
            if len(contour) < 5:
                continue
            fixcontour = [pt[0] for pt in contour] 
            fixcontour = remove_consecutive_duplicates(fixcontour) 
            rdp_points = rdp(fixcontour, epsilon=rdp_epsilon) 
            rdptotal += len(rdp_points) 

            custom_points, custom_idx = svcfp( 
                fixcontour, min_radius, max_radius, curvature_threshold, 
                rdp_epsilon, insert_threshold, fuse_radio, fuse_threshold, ifserver=0 
            ) 

            all_custom_points.extend(custom_points) 
            pointtotal += len(custom_points) 

            for i in range(len(custom_idx) - 1): 
                start_seg, end_seg = custom_idx[i], custom_idx[i + 1] 
                target_curve = np.array(fixcontour[start_seg:end_seg+1]) 
                if len(target_curve) < 5: 
                    continue 

                # 最小二乘法擬合
                try:
                    ctrl_lstsq = fit_fixed_end_bezier(target_curve) 
                    if ctrl_lstsq is not None:
                        curve_lstsq = bezier_curve_calculate(ctrl_lstsq) 
                        final_lstsq = draw_curve_on_image(final_lstsq, curve_lstsq, 1, (0, 0, 255)) 
                        only_lstsq = draw_curve_on_image(cv2.cvtColor(only_lstsq, cv2.COLOR_GRAY2BGR), curve_lstsq, 1, (0, 0, 0))
                        only_lstsq = cv2.cvtColor(only_lstsq, cv2.COLOR_BGR2GRAY)
                except Exception as e:
                    print(f"  ⚠️ LSTSQ擬合出錯: {e}")
                    continue

        t2_svcfp_lstsq = time.time()
        total_time_svcfp_lstsq = t2_svcfp_lstsq - t1_svcfp_lstsq

        # SVCFP + GA (完整流程时间测量)
        t1_svcfp_ga = time.time() 
        for contour in contours: 
            if len(contour) < 5:
                continue
            fixcontour = [pt[0] for pt in contour] 
            fixcontour = remove_consecutive_duplicates(fixcontour) 

            custom_points_ga, custom_idx_ga = svcfp( 
                fixcontour, min_radius, max_radius, curvature_threshold, 
                rdp_epsilon, insert_threshold, fuse_radio, fuse_threshold, ifserver=0 
            ) 

            for i in range(len(custom_idx_ga) - 1): 
                start_seg, end_seg = custom_idx_ga[i], custom_idx_ga[i + 1] 
                target_curve = np.array(fixcontour[start_seg:end_seg+1]) 
                if len(target_curve) < 5: 
                    continue 

                # GA 擬合 (加入超时和错误处理)
                try:
                    print(f"  🔄 GA處理第 {i+1} 段，長度 {len(target_curve)}")
                    ga_start_time = time.time()
                    ctrl_ga = genetic_algorithm(target_curve, tuple(target_curve[0]), tuple(target_curve[-1]), width, height, 30, 200, ifserver=0) 
                    ga_end_time = time.time()
                    
                    if ga_end_time - ga_start_time > 30:  # 超过30秒警告
                        print(f"  ⚠️ GA處理時間過長: {ga_end_time - ga_start_time:.2f}秒")
                    
                    if ctrl_ga is not None:
                        curve_ga = bezier_curve_calculate(ctrl_ga) 
                        if curve_ga is not None and len(curve_ga) > 0:
                            final_ga = draw_curve_on_image(final_ga, curve_ga, 1, (0, 0, 255)) 
                            only_ga = draw_curve_on_image(cv2.cvtColor(only_ga, cv2.COLOR_GRAY2BGR), curve_ga, 1, (0, 0, 0))
                            only_ga = cv2.cvtColor(only_ga, cv2.COLOR_BGR2GRAY)
                        else:
                            print(f"  ⚠️ GA生成的曲線為空")
                    else:
                        print(f"  ⚠️ GA返回控制點為None")
                        
                except Exception as e:
                    print(f"  ❌ GA擬合出錯: {e}")
                    continue

        t2_svcfp_ga = time.time()
        total_time_svcfp_ga = t2_svcfp_ga - t1_svcfp_ga

        bin_orig = cv2.threshold(orig, 127, 255, cv2.THRESH_BINARY_INV)[1] 
        bin_lstsq = cv2.threshold(cv2.cvtColor(final_lstsq, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY_INV)[1] 
        bin_ga = cv2.threshold(cv2.cvtColor(final_ga, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY_INV)[1] 

        A_lstsq = np.argwhere(bin_lstsq == 255) 
        A_ga = np.argwhere(bin_ga == 255) 
        B = np.argwhere(bin_orig == 255) 

        score_lstsq = scs_shape_similarity(A_lstsq, B) 
        score_ga = scs_shape_similarity(A_ga, B) 

        time_prefix = datetime.now().strftime("%H%M%S") 
        tag_ga = f"{time_prefix}_{score_ga:.5f}_{pointtotal}_SVCFP+GA({total_time_svcfp_ga:.2f}s)" 
        tag_lstsq = f"{time_prefix}_{score_lstsq:.5f}_{pointtotal}_SVCFP+LSTSQ({total_time_svcfp_lstsq:.2f}s)" 

        # 疊圖 
        orig_bgr = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR) 
        mask_lstsq = np.any(final_lstsq != [255, 255, 255], axis=-1) 
        mask_ga = np.any(final_ga != [255, 255, 255], axis=-1) 

        comb_lstsq = orig_bgr.copy() 
        comb_ga = orig_bgr.copy() 
        comb_lstsq[mask_lstsq] = final_lstsq[mask_lstsq] 
        comb_ga[mask_ga] = final_ga[mask_ga] 

        # 儲存圖像（白底黑線） 
        cv2.imwrite(os.path.join(image_dir, f"{tag_lstsq}_contour.png"), orig) 
        cv2.imwrite(os.path.join(image_dir, f"{tag_lstsq}_lstsq_output.png"), comb_lstsq) 
        cv2.imwrite(os.path.join(image_dir, f"{tag_lstsq}_lstsq_fitting.png"), only_lstsq) 
        cv2.imwrite(os.path.join(image_dir, f"{tag_ga}_ga_output.png"), comb_ga) 
        cv2.imwrite(os.path.join(image_dir, f"{tag_ga}_ga_fitting.png"), only_ga) 

        print(f"✅ {image_name} 處理完成 | SVCFP+LSTSQ: {score_lstsq:.4f} ({total_time_svcfp_lstsq:.2f}s), SVCFP+GA: {score_ga:.4f} ({total_time_svcfp_ga:.2f}s)") 

    except Exception as e: 
        print(f"❌ 錯誤處理圖片 {image_name}: {e}")