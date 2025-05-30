import cv2
import numpy as np

def fill_small_contours(image_path, area_threshold=1000, output_path="test/red_fill_small_regions_only.png"):
    """
    針對輸入的紅線輪廓圖像，自動填滿較小的封閉區域為紅色，保留紅色線條細節。

    參數:
        image_path (str): 輸入圖像路徑（例如紅色輪廓的 .jpg 或 .png 圖）
        area_threshold (int): 面積閾值，小於此值的輪廓才會被填色
        output_path (str): 輸出結果儲存路徑

    回傳:
        str: 輸出圖像檔案的路徑
    """
    # 讀取原圖（BGR格式）
    img = cv2.imread(image_path)

    # 灰階處理 + 二值化，取得紅線輪廓
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # 尋找所有輪廓（含層級資訊）
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 複製圖像作為結果底圖
    result = img.copy()

    # 只填小面積輪廓區域
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < area_threshold:
            cv2.drawContours(result, [contour], -1, (0, 0, 255), thickness=cv2.FILLED)

    # 保留紅線條細節（避免被填色蓋掉）
    line_mask = cv2.inRange(img, (0, 0, 100), (100, 100, 255))
    result[line_mask > 0] = [0, 0, 255]

    # 儲存輸出結果
    #cv2.imwrite(output_path, result)
    cv2.imshow(output_path,result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return output_path
    

# 範例呼叫
fill_small_contours("test/red_layer.png", area_threshold=600)