# -*- coding: utf-8 -*-
import sys
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory, Response
import time
import threading
import os
from utils import *  # 導入所有工具函數，包括 server_tools 中的函數
import base64
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
#print(os.getcwd())
#system initialization
sys.stdout.reconfigure(encoding='utf-8')  # 改變輸出的

#flask initialization
console_output = ""  # 初始化一個空字串，用於儲存 console 內容
# 創建一個列表來存儲 console_output，這樣可以通過引用來修改
console_output_ref = [console_output]
# 設置 console_output 的引用，讓 custom_print 函數可以修改它
set_console_output_ref(console_output_ref)

image_base64 = []
beizer_array = [] #this
version = "V25.4.4"


app = Flask(__name__)

@app.route('/ver.js')
def ver_js():
    global version
    js_content = f"version = '{version}';"  # 使用 f-string 格式化字串
    return Response(js_content, mimetype='application/javascript')
    
@app.route('/')
def serve_index():
    #return send_from_directory('.', 'index_ver_25.3.30.0.html')
    #版本碼不要在檔名上，直接在 index.html 程式碼內
    return send_from_directory('.', 'index.html')
    
@app.route('/message')
def get_message():
    global console_output_ref
    global image_base64
    global beizer_array
    
    message = console_output_ref[0]  # 取得當前 console_output 的內容
    console_output_ref[0] = ""       # 清空 console_output 字串

    if beizer_array:
        beizers_list = []
        for sublist in beizer_array:
            for x, y in sublist:
                beizers_list.append((int(x), int(y)))
        beizer_array=[]
        return jsonify({"message": message, "beizers": beizers_list})
        #return jsonify({"message": "123"})
           
    elif image_base64:                    #如果有圖片
         image_base64_1 = image_base64.pop(0)   #複製圖片供回覆
         return jsonify({"message": message, "imageBase64": image_base64_1})
    else:
        return jsonify({"message": message})    #無圖片，僅回應訊息
        
lock = threading.Lock()

def process_upload(width, height, contours, testmode):
    global beizer_array, image_base64

    # --- 可調參數 ---
    rdp_epsilon = 4             # RDP簡化閾值
    curvature_threshold = 41   # 曲率閾值
    min_radius = 10             # 最小搜尋半徑
    max_radius = 50             # 最大搜尋半徑
    insert_threshold=300
    fuse_radio = 8
    fuse_threshold=10
    # ----------------

    with lock:
        try:
            if testmode:
                # 白底彩色圖像：便於畫紅線與綠點
                final = np.ones((height, width, 3), dtype=np.uint8) * 255
            else:
                # 黑底圖像（不影響實際圖像輸出）
                final = np.zeros((height, width, 3), dtype=np.uint8)

            start_time = time.time()
            custom_print("Receiving contours...")
            
            rdptotal = 0
            pointtotal = 0
            result = []
            for contour in contours:
                fixcontour=interpolate_points(contour)
                rdp_points = rdp(fixcontour, epsilon=rdp_epsilon)
                rdptotal += len(rdp_points)

                # 執行特徵點擷取
                custom_points, custom_idx = svcfp(
                    fixcontour,
                    min_radius=min_radius,
                    max_radius=max_radius,
                    curvature_threshold=curvature_threshold,
                    rdp_epsilon=rdp_epsilon,
                    insert_threshold=insert_threshold,
                    fuse_radio=fuse_radio,
                    fuse_threshold=fuse_threshold,
                    ifserver=1
                )

                pointtotal += len(custom_points)
                path = fixcontour

                for i in range(len(custom_idx) - 1):
                    start = custom_idx[i]
                    end = custom_idx[i + 1]
                    target_curve = path[start:end+1]
                    print(start,end)
                    target_curve = np.array([(int(p[0]), int(p[1])) for p in target_curve])
                    print(target_curve[0],target_curve[-1])

                    if len(target_curve) == 0:
                        custom_print(f"⚠️ Line {i} 空曲線跳過")
                        continue

                    custom_print(f"Line {i}: {target_curve[0]} -> {target_curve[-1]}")

                    ctrl_pts = fit_fixed_end_bezier(target_curve)


                    result.append(ctrl_pts)
                    if not testmode:
                        beizer_array.append(ctrl_pts)

                    curve_points = bezier_curve_calculate(ctrl_pts)
                    final = draw_curve_on_image(final, curve_points, thickness=2, color=(0, 0, 255))  # 紅色線條
                    
                    if len(curve_points) == 0:
                        custom_print(f"⚠️ Line {i} 沒產生曲線點")
                    elif np.count_nonzero(cv2.cvtColor(final.copy(), cv2.COLOR_BGR2GRAY)) == 0:
                        custom_print(f"⚠️ Line {i} 畫完仍為全黑圖")
                 #存檔用
            """
            try:
                save_dir = os.path.join(os.getcwd(), "img")
                os.makedirs(save_dir, exist_ok=True)

                orig = np.zeros((height, width), dtype=np.uint8)
                for contour in contours:
                    contour = interpolate_points(contour)
                    for i in contour:
                        x, y = map(int, i)
                        if 0 <= x < width and 0 <= y < height:
                            orig[y][x] = 255

                orig = 255 - orig  # 反白處理（白底黑線）
                temp = 255 - final
                temp = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
                _, temp1 = cv2.threshold(temp, 200, 255, cv2.THRESH_BINARY)
                _, temp2 = cv2.threshold(orig, 200, 255, cv2.THRESH_BINARY)
                #showimg(orig)
                #showimg(temp)
                A = np.argwhere(temp1==0)
                B = np.argwhere(temp2==0)    
                value = scs_shape_similarity(A, B)
                print(value)

                cv2.imwrite(os.path.join(save_dir, f"{value:.5f}_{len(custom_points)}_orig.png"), orig)
                cv2.imwrite(os.path.join(save_dir, f"{value:.5f}_{len(custom_points)}_fitting.png"), temp)
                
            except Exception as e:
                print(f"❌ 存檔時發生錯誤: {e}")

                orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
                # 將 final 非白色區域覆蓋到 orig 上
                mask = np.any(final != [255, 255, 255], axis=-1)
                combined = orig.copy()
                combined[mask] = final[mask]
                final = combined
                cv2.imwrite(os.path.join(save_dir, f"{value:.5f}_{len(custom_points)}_output.png"),final)
            """

            if testmode:
                """
                # 畫綠色特徵點
                for point in rdp_points:
                    final = cv2.circle(final, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)
                """
                
                for point in custom_points:
                    final = cv2.circle(final, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)    
                end_time = time.time()
                custom_print(f"✅ 處理完成！共花費 {end_time - start_time:.2f} 秒")
                image_base64.append(encode_image_to_base64(final))
                
           
        except Exception as e:
            import traceback
            error_message = traceback.format_exc()
            print("❌ process_upload 發生錯誤：", error_message)

def process_upload_image(image_data, width, height, testmode):
    global beizer_array, image_base64
    # 這是處理 Base64 編碼圖片的函數
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
    ifshow = 0
    
    custom_print(f"處理 Canvas 圖像: width={width}, height={height}, image_data_len={len(image_data) if image_data else 0}")
    
    try:
        # 解碼 Base64 圖像數據
        # 移除 data:image/png;base64, 前綴（如果存在）
        if image_data.startswith('data:image'):
            # 找到逗號後的實際 base64 數據
            base64_data = image_data.split(',')[1]
        else:
            base64_data = image_data
        
        # 解碼 Base64
        image_bytes = base64.b64decode(base64_data)
        
        # 使用 PIL 讀取圖像
        pil_image = Image.open(BytesIO(image_bytes))
        
        # 轉換為 numpy 數組 (RGB)
        image_array = np.array(pil_image)
        
        # 如果是 RGBA，轉換為 RGB
        if image_array.shape[2] == 4:
            # 創建白色背景
            background = np.ones((image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8) * 255
            # 處理 alpha 通道
            alpha = image_array[:, :, 3] / 255.0
            for c in range(3):
                background[:, :, c] = background[:, :, c] * (1 - alpha) + image_array[:, :, c] * alpha
            image_array = background
        
        # 轉換為 BGR 格式 (OpenCV 使用 BGR)
        image_data = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # 現在 image_data 是正確的 numpy 數組，可以進行後續處理
        original_img = image_data.copy()  # 現在這行會正常工作
        gray_img = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        
        preprocessed_img = preprocess_image(gray_img, scale_factor, blur_ksize, threshold_value, ifshow)
        contours, hierarchy = cv2.findContours(preprocessed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        hierarchy = hierarchy[0]

        contours = shrink_contours(contours, final_shrink_factor)
        total_ctrl_pts = []
        hierarchy_levels = []
        contour_levels = get_contour_levels(hierarchy)

        raw_points = []       # 原始輪廓點
        fitted_points = []    # 擬合貝茲曲線點

        for contour_idx, contour in enumerate(contours):
            fixcontour = [sublist[0] for sublist in contour]
            fixcontour = remove_consecutive_duplicates(fixcontour)

            #print(fixcontour[0],fixcontour[-1])
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
            #custom_idx[-1]=custom_idx[0]

            custom_idx= remove_close_points(custom_idx,5)

            for i in range(len(custom_idx) - 1):
                start = custom_idx[i]
                end = custom_idx[i + 1]
                target_curve = path[start:end+1]
                if i==len(custom_idx) -2:
                    target_curve.append(path[0])
                target_curve = np.array([(int(p[0]), int(p[1])) for p in target_curve])
                ctrl_pts = fit_least_squares_bezier(target_curve)

                curve_pts = bezier_curve_calculate(ctrl_pts)
                fitted_points.extend(curve_pts)
                draw_curve_on_image(original_img, curve_pts)

                total_ctrl_pts.append(ctrl_pts)
                hierarchy_levels.append(contour_levels[contour_idx])
        # 閉合修正
        for i in range(len(total_ctrl_pts)):
            beizer_array.append(total_ctrl_pts[i])
        #print("B",len(beizer_array))
        #image_base64.append(encode_image_to_base64(original_img))
        custom_print("COMPLETE!")
        return True

    except Exception as e:
        import traceback
        error_message = traceback.format_exc()
        print("❌ process_upload 發生錯誤：", error_message)
        return False      
                

@app.route('/upload', methods=['POST'])
def upload():
    testmode = request.args.get('testmode')
    width = request.json.get("width")
    height = request.json.get("height")
    
    # 嘗試獲取繪圖點數據
    paths = request.json.get("points")
    
    # 嘗試獲取 Base64 編碼的圖片數據
    image_data = request.json.get("image")
    thread = None
    response_message = "Received successfully\n"

    if paths is not None and len(paths) > 0:
        # 處理繪圖點模式
        custom_print("Received paths:"+str(paths))
        thread = process_upload(width, height, paths, testmode == 'true')
        
    elif image_data is not None and len(image_data) > 0:
        # 處理 Base64 編碼圖片模式
        custom_print("Received image data (Base64).")
        
        # 將 image_data 和 width, height 傳遞給 process_upload_image
        thread = threading.Thread(target=process_upload_image, args=(image_data, width, height, testmode == 'true'))
        
    else:
        # 如果既沒有 points 也沒有 image 數據
        response_message = "No valid data (points or image) received.\n"
        custom_print(response_message)
        return jsonify({"message": response_message}), 400 # 返回 400 Bad Request
    if thread:
        thread.start()

    # 回應上傳成功
    return jsonify({"message": response_message})

if __name__ == '__main__':
    global model
    #CNN
    """
    #name = '/root/icdtgw/app/jaj/CNN_model_500_Final.h5'
    name = 'CNN_model_500_Final.h5'
    model = tf.keras.models.load_model(name,custom_objects={'euclidean_distance_loss': euclidean_distance_loss})
    """
    http_port = 41881 #for nodered1.hmi.tw
    #http_port = 32222 #for bezier.hmi.tw
    #http_port = 8000 #for localhost
    app.run(host="0.0.0.0", port=http_port, threaded=False)

