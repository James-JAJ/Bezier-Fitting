# -*- coding: utf-8 -*-
import numpy as np
import sys
import os
import cv2


sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *



if __name__ == "__main__":
    image_path = 'test/B.png'
    scale_factor = 2
    final_shrink_factor = 0.5
    blur_ksize = 3
    threshold_value = 200
    epsilon = 1.0
    rdp_epsilon = 2
    curvature_threshold = 42
    min_radius = 10
    max_radius = 50
    insert_threshold = 100
    fuse_radio = 5
    fuse_threshold = 10
    debug = True
    ifshow = 0

    try:
        original_img, gray_img = inputimg_colortogray(image_path)
        preprocessed_img = preprocess_image(gray_img, scale_factor, blur_ksize, threshold_value, ifshow)
        contours, hierarchy = cv2.findContours(preprocessed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        hierarchy = hierarchy[0]

        contours = shrink_contours(contours, final_shrink_factor)
        total_ctrl_pts = []
        hierarchy_levels = []
        contour_levels = get_contour_levels(hierarchy)

        raw_points = []       # 原始輪廓點
        fitted_points = []    # 擬合貝茲曲線點

        img = np.zeros((original_img.shape[0], original_img.shape[1], 3), dtype=np.uint8)

        for contour_idx, contour in enumerate(contours):
            fixcontour = [sublist[0] for sublist in contour]
            fixcontour = remove_consecutive_duplicates(fixcontour)
            raw_points.extend(fixcontour)

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
            path = fixcontour
            for i in range(len(custom_idx) - 1):
                start = custom_idx[i]
                end = custom_idx[i + 1]
                target_curve = path[start:end]
                target_curve = np.array([(int(p[0]), int(p[1])) for p in target_curve])
                ctrl_pts = fit_least_squares_bezier(target_curve)

                curve_pts = bezier_curve_calculate(ctrl_pts)
                fitted_points.extend(curve_pts)
                draw_curve_on_image(img, curve_pts)

                total_ctrl_pts.append(ctrl_pts)
                hierarchy_levels.append(contour_levels[contour_idx])

        # 閉合修正
        for i in range(len(total_ctrl_pts)):
            end_i = np.array(total_ctrl_pts[i][3])
            for j in range(len(total_ctrl_pts)):
                if i == j:
                    continue
                start_j = np.array(total_ctrl_pts[j][0])
                if np.linalg.norm(end_i - start_j) <= 2:
                    total_ctrl_pts[j][0] = tuple(end_i)
        print(total_ctrl_pts)
        img = fill_small_contours(img, area_threshold=3000)
        showimg(img)


    except Exception as e:
        print(f"發生錯誤: {e}")
        import traceback
        traceback.print_exc()
