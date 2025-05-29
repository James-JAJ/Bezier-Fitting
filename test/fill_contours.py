import cv2
import numpy as np
from collections import deque

def ray_casting_fill(input_img, line_thickness=2, min_area=30, debug=False):
    """
    使用射線投射算法精確計算區域嵌套層級的填充函數
    
    Args:
        input_img: 輸入圖像
        line_thickness: 線條加粗厚度
        min_area: 最小區域面積閾值
        debug: 是否顯示調試信息
    Returns:
        result_img: 填充後的圖像
    """
    
    # 轉換為灰階
    if len(input_img.shape) == 3:
        gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = input_img.copy()
    
    h, w = gray.shape
    print(f"處理圖像尺寸: {w}x{h}")
    
    # 保存原始線條
    _, original_binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    if np.mean(original_binary) > 127:
        original_binary = 255 - original_binary
    original_lines = (original_binary == 0)
    
    # 二值化並加粗線條
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    if np.mean(binary) > 127:
        binary = 255 - binary
    
    # 多步驟形態學處理，更好地封閉缺口
    kernel1 = np.ones((line_thickness, line_thickness), np.uint8)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (line_thickness+1, line_thickness+1))
    
    # 先擴張再腐蝕，封閉小缺口
    processed = cv2.dilate(binary, kernel1, iterations=1)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel2)
    
    if debug:
        cv2.imshow('Processed Binary', processed)
    
    # 使用連通組件分析檢測所有區域
    inverted = 255 - processed
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        inverted, connectivity=8)
    
    print(f"檢測到 {num_labels} 個連通組件")
    
    # 過濾有效組件
    valid_components = []
    for i in range(1, num_labels):  # 跳過背景
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            valid_components.append({
                'id': i,
                'area': area,
                'center': centroids[i],
                'bbox': (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                        stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
            })
    
    print(f"有效組件數量: {len(valid_components)}")
    
    def ray_casting_nesting_level(point, processed_binary):
        """
        使用射線投射算法計算點的嵌套層級
        從點向右發射射線，計算與邊界的交點數量
        """
        x, y = int(point[0]), int(point[1])
        if x >= w or y >= h or x < 0 or y < 0:
            return 0
        
        intersections = 0
        
        # 從當前點向右發射射線
        for check_x in range(x + 1, w):
            # 檢查是否穿越邊界（從白到黑或從黑到白）
            current_pixel = processed_binary[y, check_x]
            prev_pixel = processed_binary[y, check_x - 1]
            
            # 邊界檢測：從白色區域進入黑色線條
            if prev_pixel == 255 and current_pixel == 0:
                intersections += 1
        
        return intersections
    
    def improved_nesting_level(comp_id, components_list, labels_map, processed_binary):
        """
        改進的嵌套層級計算，結合多種方法
        """
        comp_info = next(c for c in components_list if c['id'] == comp_id)
        center = comp_info['center']
        
        # 方法1: 射線投射
        ray_level = ray_casting_nesting_level(center, processed_binary)
        
        # 方法2: 檢查包含關係
        contain_level = 0
        comp_mask = (labels_map == comp_id)
        
        for other_comp in components_list:
            if other_comp['id'] == comp_id:
                continue
                
            other_mask = (labels_map == other_comp['id'])
            
            # 檢查當前組件是否被其他組件包含
            if (other_comp['area'] > comp_info['area'] and
                other_mask[int(center[1]), int(center[0])]):
                contain_level += 1
        
        # 使用射線投射結果，但用包含關係驗證
        final_level = max(ray_level, contain_level)
        
        if debug:
            print(f"組件 {comp_id}: 射線層級={ray_level}, 包含層級={contain_level}, 最終層級={final_level}")
        
        return final_level
    
    # 計算每個組件的嵌套層級
    component_levels = {}
    for comp in valid_components:
        level = improved_nesting_level(comp['id'], valid_components, labels, processed)
        component_levels[comp['id']] = level
        print(f"組件 {comp['id']}: 面積={comp['area']:.0f}, 嵌套層級={level}")
    
    # 創建結果圖像
    result = np.ones_like(gray) * 255  # 白色背景
    
    # 按層級和面積排序填充
    sorted_components = sorted(valid_components, 
                             key=lambda x: (component_levels[x['id']], -x['area']))
    
    fill_colors = []
    for comp in sorted_components:
        comp_id = comp['id']
        level = component_levels[comp_id]
        
        # 創建組件遮罩
        comp_mask = (labels == comp_id)
        
        # 根據層級確定填充顏色
        if level % 2 == 0:
            fill_color = 255  # 偶數層白色
            color_name = "白色"
        else:
            fill_color = 0    # 奇數層黑色  
            color_name = "黑色"
        
        result[comp_mask] = fill_color
        fill_colors.append((comp_id, level, color_name, comp['area']))
        
        if debug:
            print(f"填充組件 {comp_id} (層級 {level}) - {color_name}, 面積: {comp['area']:.0f}")
    
    # 恢復原始線條
    result[original_lines] = 0
    
    print("\n=== 填充結果摘要 ===")
    for comp_id, level, color, area in fill_colors:
        print(f"組件 {comp_id}: 層級 {level} -> {color} (面積: {area:.0f})")
    
    return result

def enhanced_bucket_fill(input_img, sample_points=5, line_thickness=3, min_area=50):
    """
    增強版本：使用多點採樣提高嵌套層級檢測精度
    """
    # 轉換為灰階
    if len(input_img.shape) == 3:
        gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = input_img.copy()
    
    h, w = gray.shape
    
    # 保存原始線條
    _, original_binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    if np.mean(original_binary) > 127:
        original_binary = 255 - original_binary
    original_lines = (original_binary == 0)
    
    # 預處理
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    if np.mean(binary) > 127:
        binary = 255 - binary
    
    # 更強的形態學處理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (line_thickness, line_thickness))
    processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    processed = cv2.dilate(processed, kernel, iterations=1)
    
    # 連通組件分析
    inverted = 255 - processed
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    
    # 過濾組件
    valid_components = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            valid_components.append({
                'id': i,
                'area': area,
                'center': centroids[i],
                'mask': (labels == i)
            })
    
    print(f"檢測到 {len(valid_components)} 個有效區域")
    
    def multi_point_ray_casting(component):
        """多點射線投射，提高準確性"""
        mask = component['mask']
        
        # 獲取組件的所有像素點
        y_coords, x_coords = np.where(mask)
        
        if len(y_coords) == 0:
            return 0
        
        # 選擇多個採樣點
        sample_indices = np.linspace(0, len(y_coords)-1, min(sample_points, len(y_coords)), dtype=int)
        
        levels = []
        for idx in sample_indices:
            y, x = y_coords[idx], x_coords[idx]
            
            # 射線投射
            intersections = 0
            for check_x in range(x + 1, w):
                if processed[y, check_x] == 0 and processed[y, check_x-1] == 255:
                    intersections += 1
            
            levels.append(intersections)
        
        # 返回最常見的層級
        if levels:
            return max(set(levels), key=levels.count)
        return 0
    
    # 計算層級
    component_levels = {}
    for comp in valid_components:
        level = multi_point_ray_casting(comp)
        component_levels[comp['id']] = level
        print(f"組件 {comp['id']}: 面積={comp['area']}, 層級={level}")
    
    # 填充結果
    result = np.ones_like(gray) * 255
    
    for comp in valid_components:
        comp_id = comp['id']
        level = component_levels[comp_id]
        
        if level % 2 == 1:  # 奇數層填黑色
            result[comp['mask']] = 0
            print(f"填充組件 {comp_id} (層級 {level}) - 黑色")
        else:  # 偶數層保持白色
            print(f"保持組件 {comp_id} (層級 {level}) - 白色")
    
    # 恢復原始線條
    result[original_lines] = 0
    
    return result

# 使用範例
if __name__ == "__main__":
    # 讀取圖像
    img = cv2.imread('test/red_layer.png')
    
    # 方法1: 射線投射填充（推薦）
    result1 = ray_casting_fill(img, line_thickness=2, min_area=50, debug=True)
    
    # 方法2: 增強桶填充
    result2 = enhanced_bucket_fill(img, sample_points=7, line_thickness=5, min_area=30)
    
    # 顯示結果
    cv2.imshow('Original', img)
    cv2.imshow('Ray Casting Result', result1)
    cv2.imshow('Enhanced Bucket Fill Result', result2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    pass