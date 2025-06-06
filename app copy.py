# -*- coding: utf-8 -*-
import sys
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory, Response
import time
import threading
import os
from datetime import datetime
from utils import *  # 請確保 utils.py 中包含所需的函數

# 設置標準輸出編碼
sys.stdout.reconfigure(encoding='utf-8')

# 全局變數初始化
console_output = ""
console_output_ref = [console_output]
set_console_output_ref(console_output_ref)

image_base64 = []
beizer_array = []
version = "V25.4.4"

# 創建 Flask 應用
app = Flask(__name__)

@app.route('/ver.js')
def ver_js():
    """返回版本信息的 JavaScript 文件"""
    global version
    js_content = f"version = '{version}';"
    return Response(js_content, mimetype='application/javascript')

@app.route('/')
def serve_index():
    """提供主頁面"""
    return send_from_directory('.', 'index.html')

@app.route('/message')
def get_message():
    """獲取處理消息和結果"""
    global console_output_ref, image_base64, beizer_array
    
    # 獲取並清空消息
    message = console_output_ref[0]
    console_output_ref[0] = ""
    
    # 準備返回數據
    response_data = {"message": message}
    
    # 檢查是否有 Bezier 曲線數據
    if beizer_array:
        beizers_list = [(int(x), int(y)) for x, y in beizer_array.pop(0)]
        response_data["beizers"] = beizers_list
    
    # 檢查是否有圖像數據
    elif image_base64:
        response_data["imageBase64"] = image_base64.pop(0)
    
    return jsonify(response_data)

# 線程鎖，確保並發安全
lock = threading.Lock()

def process_upload(width, height, contours, testmode):
    """
    處理上傳的輪廓數據並進行 Bezier 曲線擬合
    
    Args:
        width (int): 圖像寬度
        height (int): 圖像高度
        contours (list): 輪廓點列表
        testmode (bool): 是否為測試模式
    """
    global beizer_array, image_base64
    
    # 參數配置
    rdp_epsilon = 4
    curvature_threshold = 41
    min_radius = 10
    max_radius = 50
    insert_threshold = 300
    fuse_radio = 8
    fuse_threshold = 10
    
    with lock:
        try:
            # 初始化結果圖像
            background_color = 255 if testmode else 0
            ga_result = np.ones((height, width, 3), dtype=np.uint8) * background_color
            lstsq_result = np.ones((height, width, 3), dtype=np.uint8) * background_color
            final_ga = ga_result.copy()
            final_lstsq = lstsq_result.copy()
            
            start_time = time.time()
            custom_print("開始處理輪廓數據...")
            
            # 統計變量
            rdptotal = 0
            pointtotal = 0
            result = []
            all_custom_points = []
            total_time_lstsq = 0
            total_time_ga = 0
            
            # 處理每個輪廓
            for idx, contour in enumerate(contours):
                custom_print(f"處理輪廓 {idx + 1}/{len(contours)}")
                
                # 插值和簡化輪廓點
                fixcontour = interpolate_points(contour)
                rdp_points = rdp(fixcontour, epsilon=rdp_epsilon)
                rdptotal += len(rdp_points)
                
                # 獲取特徵點
                custom_points, custom_idx = svcfp(
                    fixcontour, min_radius, max_radius, curvature_threshold,
                    rdp_epsilon, insert_threshold, fuse_radio, fuse_threshold, 
                    ifserver=1
                )
                
                all_custom_points.extend(custom_points)
                pointtotal += len(custom_points)
                path = fixcontour
                
                # 為每個線段進行 Bezier 曲線擬合
                for i in range(len(custom_idx) - 1):
                    start_seg, end_seg = custom_idx[i], custom_idx[i + 1]
                    target_curve = np.array([
                        (int(p[0]), int(p[1])) for p in path[start_seg:end_seg+1]
                    ])
                    
                    if len(target_curve) == 0:
                        continue
                    
                    # 最小二乘法擬合
                    t1 = time.time()
                    ctrl_pts_lstsq = fit_fixed_end_bezier(target_curve)
                    t2 = time.time()
                    total_time_lstsq += t2 - t1
                    
                    # 遺傳算法擬合
                    t3 = time.time()
                    ctrl_pts_ga = genetic_algorithm(
                        target_curve, target_curve[0], target_curve[-1], 
                        width, height, 30, 200, 1
                    )
                    t4 = time.time()
                    total_time_ga += t4 - t3
                    
                    # 保存結果
                    result.append(ctrl_pts_lstsq)
                    if not testmode:
                        beizer_array.append(ctrl_pts_lstsq)
                    
                    # 繪製曲線
                    curve_lstsq = bezier_curve_calculate(ctrl_pts_lstsq)
                    curve_ga = bezier_curve_calculate(ctrl_pts_ga)
                    
                    final_lstsq = draw_curve_on_image(final_lstsq, curve_lstsq, 1, (0, 0, 255))
                    final_ga = draw_curve_on_image(final_ga, curve_ga, 1, (0, 0, 255))
            
            # 創建保存目錄
            save_dir = os.path.join(os.getcwd(), "img")
            os.makedirs(save_dir, exist_ok=True)
            
            # 創建原始圖像
            orig = np.zeros((height, width), dtype=np.uint8)
            for contour in contours:
                for point in interpolate_points(contour):
                    x, y = map(int, point)
                    if 0 <= x < width and 0 <= y < height:
                        orig[y][x] = 255
            
            # 圖像處理和相似度計算
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
            
            # 生成文件名標籤
            time_prefix = datetime.now().strftime("%H%M%S")
            lstsq_tag = f"{time_prefix}_{score_lstsq:.5f}_{pointtotal}_({total_time_lstsq:.2f}s)"
            ga_tag = f"{time_prefix}_{score_ga:.5f}_{pointtotal}_({total_time_ga:.2f}s)"
            
            # 保存圖像
            cv2.imwrite(os.path.join(save_dir, f"{lstsq_tag}_orig.png"), orig_inv)
            cv2.imwrite(os.path.join(save_dir, f"{lstsq_tag}_lstsq_fitting.png"), temp_lstsq)
            cv2.imwrite(os.path.join(save_dir, f"{ga_tag}_ga_fitting.png"), temp_ga)
            
            # 在圖像上標記特徵點
            orig_bgr = cv2.cvtColor(orig_inv, cv2.COLOR_GRAY2BGR)
            for pt in all_custom_points:
                final_lstsq = cv2.circle(final_lstsq, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)
                final_ga = cv2.circle(final_ga, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)
            
            # 創建組合圖像
            comb_lstsq = orig_bgr.copy()
            comb_ga = orig_bgr.copy()
            
            mask_lstsq = np.any(final_lstsq != [255, 255, 255], axis=-1)
            mask_ga = np.any(final_ga != [255, 255, 255], axis=-1)
            
            comb_lstsq[mask_lstsq] = final_lstsq[mask_lstsq]
            comb_ga[mask_ga] = final_ga[mask_ga]
            
            # 保存最終結果
            cv2.imwrite(os.path.join(save_dir, f"{lstsq_tag}_lstsq_output.png"), comb_lstsq)
            cv2.imwrite(os.path.join(save_dir, f"{ga_tag}_ga_output.png"), comb_ga)
            
            # 測試模式下返回預覽圖像
            if testmode:
                image_base64.clear()
                image_base64.append(encode_image_to_base64(comb_ga))
            
            # 處理完成提示
            total_time = time.time() - start_time
            custom_print(f"處理完成！總耗時: {total_time:.2f}s")
            custom_print(f"LSTSQ 相似度: {score_lstsq:.5f}, GA 相似度: {score_ga:.5f}")
            custom_print(f"特徵點總數: {pointtotal}, RDP 點總數: {rdptotal}")
            
        except Exception as e:
            import traceback
            error_msg = f"❌ process_upload 發生錯誤: {str(e)}"
            custom_print(error_msg)
            print("詳細錯誤信息:", traceback.format_exc())

@app.route('/upload', methods=['POST'])
def upload():
    """處理上傳請求"""
    try:
        # 獲取參數
        testmode = request.args.get('testmode', 'false').lower() == 'true'
        
        if not request.json:
            return jsonify({"error": "No JSON data received"}), 400
        
        width = request.json.get("width")
        height = request.json.get("height")
        paths = request.json.get("points")
        image_data = request.json.get("image")
        
        # 驗證必要參數
        if not width or not height:
            return jsonify({"error": "Width and height are required"}), 400
        
        if paths:
            # 在新線程中處理輪廓數據
            thread = threading.Thread(
                target=process_upload,
                args=(width, height, paths, testmode)
            )
            thread.daemon = True
            thread.start()
            
            return jsonify({"message": "輪廓數據接收成功，正在處理中...\n"})
            
        elif image_data:
            custom_print("接收到圖像數據，暫時跳過處理")
            return jsonify({"message": "圖像數據接收成功\n"})
            
        else:
            return jsonify({"error": "No valid data (points or image) received"}), 400
            
    except Exception as e:
        error_msg = f"上傳處理錯誤: {str(e)}"
        custom_print(error_msg)
        return jsonify({"error": error_msg}), 500

@app.route('/status')
def status():
    """獲取服務器狀態"""
    return jsonify({
        "status": "running",
        "version": version,
        "pending_beizers": len(beizer_array),
        "pending_images": len(image_base64)
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "API endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    http_port = 41881
    print(f"🚀 啟動 Bezier 曲線擬合服務器 {version}")
    print(f"📡 服務器地址: http://0.0.0.0:{http_port}")
    print(f"🔧 調試模式: {'開啟' if app.debug else '關閉'}")
    
    try:
        app.run(host="0.0.0.0", port=http_port, threaded=True, debug=False)
    except KeyboardInterrupt:
        print("\n⏹️  服務器已停止")
    except Exception as e:
        print(f"❌ 服務器啟動失败: {e}")