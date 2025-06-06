# -*- coding: utf-8 -*-
import sys
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory, Response
import time
import threading
import os
from datetime import datetime
from utils import *


sys.stdout.reconfigure(encoding='utf-8')

console_output = ""
console_output_ref = [console_output]
set_console_output_ref(console_output_ref)

image_base64 = []
beizer_array = []
version = "V25.4.4"

app = Flask(__name__)

@app.route('/ver.js')
def ver_js():
    global version
    js_content = f"version = '{version}';"
    return Response(js_content, mimetype='application/javascript')

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/message')
def get_message():
    global console_output_ref, image_base64, beizer_array
    message = console_output_ref[0]
    console_output_ref[0] = ""

    if beizer_array:
        beizers_list = [(int(x), int(y)) for x, y in beizer_array.pop(0)]
        return jsonify({"message": message, "beizers": beizers_list})
    elif image_base64:
        return jsonify({"message": message, "imageBase64": image_base64.pop(0)})
    else:
        return jsonify({"message": message})

lock = threading.Lock()

def process_upload(width, height, contours, testmode):
    global beizer_array, image_base64

    rdp_epsilon = 4
    curvature_threshold = 41
    min_radius = 10
    max_radius = 50
    insert_threshold = 300
    fuse_radio = 8
    fuse_threshold = 10

    with lock:
        try:
            ga_result = np.ones((height, width, 3), dtype=np.uint8) * (255 if testmode else 0)
            lstsq_result = np.ones((height, width, 3), dtype=np.uint8) * (255 if testmode else 0)
            final_ga = ga_result.copy()
            final_lstsq = lstsq_result.copy()

            start_time = time.time()
            custom_print("Receiving contours...")

            rdptotal = 0
            pointtotal = 0
            result = []
            all_custom_points = []

            total_time_lstsq = 0
            total_time_ga = 0

            for contour in contours:
                fixcontour = interpolate_points(contour)
                rdp_points = rdp(fixcontour, epsilon=rdp_epsilon)
                rdptotal += len(rdp_points)

                custom_points, custom_idx = svcfp(
                    fixcontour, min_radius, max_radius, curvature_threshold,
                    rdp_epsilon, insert_threshold, fuse_radio, fuse_threshold, ifserver=0
                )

                all_custom_points.extend(custom_points)
                pointtotal += len(custom_points)
                path = fixcontour

                for i in range(len(custom_idx) - 1):
                    start_seg, end_seg = custom_idx[i], custom_idx[i + 1]
                    target_curve = np.array([(int(p[0]), int(p[1])) for p in path[start_seg:end_seg+1]])
                    if len(target_curve) == 0:
                        continue

                    t1 = time.time()
                    ctrl_pts_lstsq = fit_fixed_end_bezier(target_curve)
                    t2 = time.time()
                    total_time_lstsq += t2 - t1

                    t3 = time.time()
                    ctrl_pts_ga = genetic_algorithm(
                        target_curve, target_curve[0], target_curve[-1], width, height, 30, 200, 0
                    )
                    t4 = time.time()
                    total_time_ga += t4 - t3

                    result.append(ctrl_pts_lstsq)
                    if not testmode:
                        beizer_array.append(ctrl_pts_lstsq)

                    curve_lstsq = bezier_curve_calculate(ctrl_pts_lstsq)
                    curve_ga = bezier_curve_calculate(ctrl_pts_ga)

                    final_lstsq = draw_curve_on_image(final_lstsq, curve_lstsq, 1, (0, 0, 255))
                    final_ga = draw_curve_on_image(final_ga, curve_ga, 1, (0, 0, 255))

            save_dir = os.path.join(os.getcwd(), "img")
            os.makedirs(save_dir, exist_ok=True)

            orig = np.zeros((height, width), dtype=np.uint8)
            for contour in contours:
                for i in interpolate_points(contour):
                    x, y = map(int, i)
                    if 0 <= x < width and 0 <= y < height:
                        orig[y][x] = 255

            orig_inv = 255 - orig
            temp_lstsq = cv2.cvtColor(final_lstsq, cv2.COLOR_BGR2GRAY)
            temp_ga = cv2.cvtColor(final_ga, cv2.COLOR_BGR2GRAY)

            _, bin_lstsq = cv2.threshold(temp_lstsq, 200, 255, cv2.THRESH_BINARY)
            _, bin_ga = cv2.threshold(temp_ga, 200, 255, cv2.THRESH_BINARY)
            _, bin_orig = cv2.threshold(orig_inv, 200, 255, cv2.THRESH_BINARY)

            A_lstsq = np.argwhere(bin_lstsq == 0)
            A_ga = np.argwhere(bin_ga == 0)
            B = np.argwhere(bin_orig == 0)

            score_lstsq = scs_shape_similarity(A_lstsq, B)
            score_ga = scs_shape_similarity(A_ga, B)

            time_prefix = datetime.now().strftime("%H%M%S")
            lstsq_tag = f"{time_prefix}_{score_lstsq:.5f}_{pointtotal}_({total_time_lstsq:.2f}s)"
            ga_tag = f"{time_prefix}_{score_ga:.5f}_{pointtotal}_({total_time_ga:.2f}s)"

            cv2.imwrite(os.path.join(save_dir, f"{lstsq_tag}_orig.png"), orig_inv)
            cv2.imwrite(os.path.join(save_dir, f"{lstsq_tag}_lstsq_fitting.png"), temp_lstsq)
            cv2.imwrite(os.path.join(save_dir, f"{ga_tag}_ga_fitting.png"), temp_ga)

            orig_bgr = cv2.cvtColor(orig_inv, cv2.COLOR_GRAY2BGR)
            for pt in all_custom_points:
                final_lstsq = cv2.circle(final_lstsq, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)
                final_ga = cv2.circle(final_ga, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)

            comb_lstsq = orig_bgr.copy()
            comb_ga = orig_bgr.copy()
            comb_lstsq[np.any(final_lstsq != [255, 255, 255], axis=-1)] = final_lstsq[np.any(final_lstsq != [255, 255, 255], axis=-1)]
            comb_ga[np.any(final_ga != [255, 255, 255], axis=-1)] = final_ga[np.any(final_ga != [255, 255, 255], axis=-1)]

            cv2.imwrite(os.path.join(save_dir, f"{lstsq_tag}_lstsq_output.png"), comb_lstsq)
            cv2.imwrite(os.path.join(save_dir, f"{ga_tag}_ga_output.png"), comb_ga)

            if testmode:
                image_base64.clear()  # 確保只保留當次結果，避免殘留
                image_base64.append(encode_image_to_base64(comb_ga))  # 或 comb_lstsq 視前端預覽需求


        except Exception as e:
            import traceback
            print("❌ process_upload 發生錯誤：", traceback.format_exc())

@app.route('/upload', methods=['POST'])
def upload():
    testmode = request.args.get('testmode')
    width = request.json.get("width")
    height = request.json.get("height")
    paths = request.json.get("points")
    image_data = request.json.get("image")

    if paths:
        thread = threading.Thread(target=process_upload, args=(width, height, paths, testmode == 'true'))
    elif image_data:
        print("SKIP")
    else:
        return jsonify({"message": "No valid data (points or image) received."}), 400

    thread.start()
    return jsonify({"message": "Received successfully\n"})

if __name__ == '__main__':
    http_port = 41881
    app.run(host="0.0.0.0", port=http_port, threaded=False)
