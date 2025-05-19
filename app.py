# -*- coding: utf-8 -*-
import sys
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory, Response
import time
import threading
import os
from utils import *  # 導入所有工具函數，包括 server_tools 中的函數
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
beizer_array = []
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
        for x, y in beizer_array.pop(0):
            beizers_list.append((int(x), int(y)))
        return jsonify({"message": message, "beizers": beizers_list})   
    elif image_base64:                    #如果有圖片
         image_base64_1 = image_base64.pop(0)   #複製圖片供回覆
         return jsonify({"message": message, "imageBase64": image_base64_1})
    else:
        return jsonify({"message": message})    #無圖片，僅回應訊息
        
lock = threading.Lock()

#處理軌跡上傳資訊 
def process_upload(width, height, paths, testmode):
    global beizer_array, image_base64
    with lock:
        try:
            # paths: 多筆畫陣列 [[點位1, 點位2, ...], [點位1, 點位2, ...], ...]
            # 每個點位是 (x, y) 座標

            final = np.zeros((height, width), dtype=np.uint8)
            start_time = time.time()
            paths = np.array(paths)  # Use dtype=object for jagged arrays
            pathslen = len(paths)
            custom_print("Receiving image...")
            custom_print("Unpoccessed paths:", pathslen)

            paths=interpolate_points(paths[0])

            custom_print("補點後",len(paths))

            # [筆畫][每筆畫點位]
            jointsrdp = rdp(paths,10)
            custom_print("rdp:", len(jointsrdp))

            joints,joint_idx = svcfp(paths,rdp_epsilon=10)
            
            result = []  # 這裡應該儲存所有路徑的控制點
            for i in range(len(joint_idx)-1):
                target_curve = []
                # 收集從 jointsA[h][i] 到 jointsA[h][i+1] 之間的所有點位
                custom_print("!",joint_idx[i], joint_idx[i+1])  
                # 收集該範圍內的所有點
                for j in range(joint_idx[i], joint_idx[i+1]+1):
                    point = paths[j]
                    
                    # 確保點位是整數
                    target_curve.append((int(point[0]), int(point[1])))
                
                # 輸出首尾點以便偵錯
                custom_print(f"Line {i} from {target_curve[0]} to {target_curve[-1]}")
                
                pre = genetic_algorithm(target_curve, target_curve[0], target_curve[-1], width, height)
                result.append(pre)
                if not testmode:
                    beizer_array.append(pre)

                # 生成貝茲曲線並繪製
                curve_points = bezier_curve_calculate(pre)
                predict = np.zeros((height, width), dtype=np.uint8)
                predict = draw_curve_on_image(predict, curve_points, 3)
                final = stack_image(final, predict)
        
            if testmode:    #單筆繪圖模式 存入 圖片
                final = cv2.cvtColor(final.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                mask_white = (final == np.array([255, 255, 255])).all(axis=2)
                mask_black = (final == np.array([0, 0, 0])).all(axis=2)
                final[mask_white] = [0, 0, 255]
                final[mask_black] = [255, 255, 255]
                """for point in jointsrdp:
                    final = cv2.circle(final, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)
                """
                
                for point in joints:
                    final = cv2.circle(final, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
                
                """
                for point in paths:
                    final = cv2.circle(final, (int(point[0]), int(point[1])), 5, (0, 255, 0), 2) 
                """
                custom_print(result)
                # 計算和顯示處理時間
                end_time = time.time()
                custom_print(f"Processing completed in {end_time - start_time:.2f} seconds")
                
                image_base64.append(encode_image_to_base64(final))
                #return result  # 返回所有線段的控制點，以便後續使用
                
        
        except Exception as e:
            import traceback
            error_message = traceback.format_exc()
            custom_print("Error in process_upload:" , error_message) #紀錄錯誤訊息
            #return []

@app.route('/upload', methods=['POST'])
def upload():
    testmode = request.args.get('testmode')
    width, height = request.json.get("width"), request.json.get("height")
    #上傳 軌跡的模式
    paths = request.json.get("points")
    #custom_print("Received paths:", paths)
    #開始擬和程式執行緒(原本單執行續方式 /message 無法即時傳送過程資訊
    thread = threading.Thread(target=process_upload,args=(width,height,paths,testmode=='true'))
    thread.start()
    #回應上傳成功
    return jsonify({"message": "Received successfully\n"})

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