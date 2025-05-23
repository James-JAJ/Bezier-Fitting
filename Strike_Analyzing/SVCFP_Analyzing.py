import cv2
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import warnings
warnings.filterwarnings('ignore')
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

class ImageAlignment:
    def __init__(self):
        self.methods = {
            'RANSAC_Affine': self.ransac_affine_alignment,
            'EMD': self.emd_alignment,
            'CPD': self.cpd_alignment,
            'Shape_Context': self.shape_context_alignment
        }
    
    def load_and_preprocess(self, img_path):
        """載入並預處理圖片"""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"無法載入圖片: {img_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, gray
    
    def extract_features(self, gray_img, method='ORB'):
        """提取特徵點"""
        if method == 'ORB':
            detector = cv2.ORB_create(nfeatures=1000)
        elif method == 'SIFT':
            detector = cv2.SIFT_create()
        
        keypoints, descriptors = detector.detectAndCompute(gray_img, None)
        
        # 轉換為點座標
        points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        return points, descriptors, keypoints
    
    def extract_contour_points(self, gray_img, num_points=100):
        """提取輪廓點用於形狀匹配"""
        # 邊緣檢測
        edges = cv2.Canny(gray_img, 50, 150)
        
        # 找輪廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # 如果沒有找到輪廓，使用網格點
            h, w = gray_img.shape
            grid_size = int(np.sqrt(num_points))
            x = np.linspace(10, w-10, grid_size)
            y = np.linspace(10, h-10, grid_size)
            xx, yy = np.meshgrid(x, y)
            points = np.column_stack([xx.ravel(), yy.ravel()])
            return points[:num_points].astype(np.float64)
        
        # 取最大輪廓
        largest_contour = max(contours, key=cv2.contourArea)
        contour_points = largest_contour.reshape(-1, 2).astype(np.float64)
        
        # 重採樣到指定數量的點
        if len(contour_points) > num_points:
            indices = np.linspace(0, len(contour_points)-1, num_points, dtype=int)
            contour_points = contour_points[indices]
        elif len(contour_points) < num_points:
            # 如果點數不足，進行插值
            from scipy.interpolate import interp1d
            
            # 計算累積弧長
            distances = np.sqrt(np.sum(np.diff(contour_points, axis=0)**2, axis=1))
            cumulative_distances = np.concatenate([[0], np.cumsum(distances)])
            
            # 創建插值函數
            if len(cumulative_distances) > 1:
                interp_x = interp1d(cumulative_distances, contour_points[:, 0], kind='linear', 
                                  bounds_error=False, fill_value='extrapolate')
                interp_y = interp1d(cumulative_distances, contour_points[:, 1], kind='linear',
                                  bounds_error=False, fill_value='extrapolate')
                
                # 生成新的採樣點
                new_distances = np.linspace(0, cumulative_distances[-1], num_points)
                new_x = interp_x(new_distances)
                new_y = interp_y(new_distances)
                contour_points = np.column_stack([new_x, new_y])
        
        return contour_points.astype(np.float64)
    
    def ransac_affine_alignment(self, target_img, reference_img):
        """RANSAC + Affine 非剛性對齊"""
        print("執行 RANSAC + Affine 對齊...")
        
        # 提取特徵點
        target_points, target_desc, _ = self.extract_features(target_img)
        ref_points, ref_desc, _ = self.extract_features(reference_img)
        
        if target_desc is None or ref_desc is None:
            raise ValueError("無法提取足夠的特徵點")
        
        # 特徵匹配
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(target_desc, ref_desc)
        matches = sorted(matches, key=lambda x: x.distance)
        
        if len(matches) < 10:
            raise ValueError("匹配點數量不足")
        
        # 提取匹配點對
        src_pts = np.float32([target_points[m.queryIdx] for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([ref_points[m.trainIdx] for m in matches]).reshape(-1, 1, 2)
        
        # 使用RANSAC估計仿射變換矩阵
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is None:
            # 如果Homography失敗，使用Affine變換
            M = cv2.getAffineTransform(src_pts[:3].reshape(-1, 2), dst_pts[:3].reshape(-1, 2))
            # 轉換為3x3矩陣
            M = np.vstack([M, [0, 0, 1]])
        
        # 應用變換
        h, w = reference_img.shape
        aligned_img = cv2.warpPerspective(target_img, M, (w, h))
        
        return aligned_img
    
    def emd_alignment(self, target_img, reference_img):
        """EMD (Earth Mover's Distance) 對齊 - 改進版"""
        print("執行 EMD 對齊...")
        
        # 提取更多的特徵點以獲得更好的對齊效果
        target_points = self.extract_dense_points(target_img, 100)
        ref_points = self.extract_dense_points(reference_img, 100)
        
        # 確保點數相同
        min_points = min(len(target_points), len(ref_points))
        target_points = target_points[:min_points]
        ref_points = ref_points[:min_points]
        
        # 計算距離矩陣
        cost_matrix = cdist(target_points, ref_points, metric='euclidean')
        
        # 使用匈牙利算法求解最小代價匹配（EMD的近似）
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # 建立對應關係
        matched_target = target_points[row_ind]
        matched_ref = ref_points[col_ind]
        
        # 使用改進的變換方法
        aligned_img = self.improved_transform(target_img, matched_target, matched_ref)
        
        return aligned_img
    
    def extract_dense_points(self, gray_img, num_points=100):
        """提取密集特徵點 - 結合輪廓點和網格點"""
        h, w = gray_img.shape
        points_list = []
        
        # 1. 提取輪廓點
        edges = cv2.Canny(gray_img, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 取最大的幾個輪廓
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
            for contour in contours:
                contour_points = contour.reshape(-1, 2).astype(np.float64)
                if len(contour_points) > 10:
                    # 均勻採樣輪廓點
                    indices = np.linspace(0, len(contour_points)-1, min(30, len(contour_points)), dtype=int)
                    points_list.extend(contour_points[indices])
        
        # 2. 添加角點
        corners = cv2.goodFeaturesToTrack(gray_img, maxCorners=50, qualityLevel=0.01, minDistance=10)
        if corners is not None:
            points_list.extend(corners.reshape(-1, 2))
        
        # 3. 添加網格點作為補充
        grid_size = int(np.sqrt(num_points // 2))
        x = np.linspace(w*0.1, w*0.9, grid_size)
        y = np.linspace(h*0.1, h*0.9, grid_size)
        xx, yy = np.meshgrid(x, y)
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        points_list.extend(grid_points)
        
        # 轉換並限制點數
        if points_list:
            all_points = np.array(points_list, dtype=np.float64)
            # 去重
            unique_points = []
            for point in all_points:
                is_duplicate = False
                for existing in unique_points:
                    if np.linalg.norm(point - existing) < 5:  # 距離閾值
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_points.append(point)
            
            unique_points = np.array(unique_points)
            if len(unique_points) > num_points:
                indices = np.random.choice(len(unique_points), num_points, replace=False)
                unique_points = unique_points[indices]
            
            return unique_points
        else:
            # 備用網格點
            x = np.linspace(10, w-10, int(np.sqrt(num_points)))
            y = np.linspace(10, h-10, int(np.sqrt(num_points)))
            xx, yy = np.meshgrid(x, y)
            return np.column_stack([xx.ravel(), yy.ravel()])[:num_points]
    
    def cpd_alignment(self, target_img, reference_img):
        """Coherent Point Drift (CPD) 對齊 - 改進版"""
        print("執行 CPD 對齊...")
        
        # 提取密集特徵點
        target_points = self.extract_dense_points(target_img, 120)
        ref_points = self.extract_dense_points(reference_img, 120)
        
        # CPD算法的簡化實現
        try:
            aligned_points = self.cpd_register(target_points, ref_points)
            
            # 使用改進的變換方法，避免空洞
            aligned_img = self.improved_transform(target_img, target_points, aligned_points)
            
            return aligned_img
            
        except Exception as e:
            print(f"CPD配準失敗，使用備用方法: {e}")
            # 備用方案：使用簡單的點對點匹配
            min_points = min(len(target_points), len(ref_points))
            target_points = target_points[:min_points]  
            ref_points = ref_points[:min_points]
            return self.improved_transform(target_img, target_points, ref_points)
    
    def cpd_register(self, X, Y, max_iter=50, tol=1e-3):
        """CPD點集配準的簡化實現"""
        X = np.array(X, dtype=np.float64)
        Y = np.array(Y, dtype=np.float64)
        
        # 初始化參數
        M, D = X.shape
        N, _ = Y.shape
        
        # 如果點數差異太大，進行重採樣
        if abs(M - N) > min(M, N) * 0.5:
            min_points = min(M, N)
            if M > min_points:
                indices = np.linspace(0, M-1, min_points, dtype=int)
                X = X[indices]
                M = min_points
            if N > min_points:
                indices = np.linspace(0, N-1, min_points, dtype=int)
                Y = Y[indices]
                N = min_points
        
        # 初始化變換參數
        W = np.zeros((M, D), dtype=np.float64)
        beta = 2.0
        lambda_reg = 3.0
        
        # 主要的CPD迭代
        for iteration in range(max_iter):
            try:
                # 計算高斯混合模型的權重
                dist_matrix = cdist(X + W, Y)
                G = np.exp(-dist_matrix**2 / (2 * beta**2))
                
                # 歸一化
                G_sum = G.sum(axis=1, keepdims=True) + 1e-8
                P = G / G_sum
                
                # 更新W
                A = P.sum(axis=1)
                
                # 避免奇異矩陣
                A_diag = np.diag(A + 1e-8)
                reg_term = lambda_reg * np.eye(M)
                
                # 求解線性系統
                lhs = A_diag + reg_term
                rhs = P @ Y - A_diag @ X
                
                try:
                    W_new = np.linalg.solve(lhs, rhs)
                except:
                    # 如果求解失敗，使用最小二乘法
                    W_new = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
                
                # 檢查收斂
                if np.linalg.norm(W_new - W) < tol:
                    break
                    
                W = W_new
                
            except Exception as e:
                print(f"CPD迭代 {iteration} 失敗: {e}")
                break
        
        return X + W
    
    def shape_context_alignment(self, target_img, reference_img):
        """Shape Context Matching 對齊 - 改進版"""
        print("執行 Shape Context 對齊...")
        
        # 提取密集特徵點
        target_points = self.extract_dense_points(target_img, 80)
        ref_points = self.extract_dense_points(reference_img, 80)
        
        # 確保點數相同
        min_points = min(len(target_points), len(ref_points))
        target_points = target_points[:min_points]
        ref_points = ref_points[:min_points]
        
        try:
            # 計算形狀上下文
            target_sc = self.compute_shape_context(target_points)
            ref_sc = self.compute_shape_context(ref_points)
            
            # 檢查形狀上下文是否有效
            if np.all(target_sc == 0) or np.all(ref_sc == 0):
                print("警告：形狀上下文計算無效，使用直接點匹配")
                # 使用距離直接匹配
                cost_matrix = cdist(target_points, ref_points, metric='euclidean')
            else:
                # 匹配形狀上下文
                cost_matrix = self.shape_context_cost(target_sc, ref_sc)
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # 建立對應關係
            matched_target = target_points[row_ind]
            matched_ref = ref_points[col_ind]
            
            # 使用改進的變換方法
            aligned_img = self.improved_transform(target_img, matched_target, matched_ref)
            
            return aligned_img
            
        except Exception as e:
            print(f"Shape Context計算失敗，使用簡單匹配: {e}")
            # 備用方案：使用距離匹配
            cost_matrix = cdist(target_points, ref_points, metric='euclidean')
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matched_target = target_points[row_ind]
            matched_ref = ref_points[col_ind]
            return self.improved_transform(target_img, matched_target, matched_ref)
    
    def compute_shape_context(self, points, n_bins_r=5, n_bins_theta=12):
        """計算形狀上下文描述子"""
        points = np.array(points, dtype=np.float64)
        n_points = len(points)
        shape_contexts = np.zeros((n_points, n_bins_r * n_bins_theta), dtype=np.float64)
        
        for i, point in enumerate(points):
            # 計算到其他所有點的相對位置
            diff = points - point
            
            # 轉為極坐標
            r = np.sqrt(diff[:, 0]**2 + diff[:, 1]**2)
            theta = np.arctan2(diff[:, 1], diff[:, 0])
            
            # 移除自己（使用布爾索引時要小心）
            valid_mask = r > 1e-10  # 使用小的閾值而不是0
            r_valid = r[valid_mask]
            theta_valid = theta[valid_mask]
            
            if len(r_valid) == 0:
                continue
            
            # 對數半徑分箱
            r_max = np.max(r_valid)
            r_min = max(np.min(r_valid), r_max/1000)  # 避免除零
            
            try:
                r_bins = np.logspace(np.log10(r_min), np.log10(r_max), n_bins_r+1)
            except:
                # 如果logspace失敗，使用線性分箱
                r_bins = np.linspace(r_min, r_max, n_bins_r+1)
            
            # 角度分箱
            theta_bins = np.linspace(-np.pi, np.pi, n_bins_theta+1)
            
            # 創建直方圖
            try:
                hist, _, _ = np.histogram2d(r_valid, theta_valid, bins=[r_bins, theta_bins])
                shape_contexts[i] = hist.flatten()
            except Exception as e:
                print(f"警告：點 {i} 的形狀上下文計算失敗: {e}")
                continue
        
        return shape_contexts
    
    def shape_context_cost(self, sc1, sc2):
        """計算形狀上下文之間的代價"""
        # 使用卡方距離
        cost_matrix = np.zeros((len(sc1), len(sc2)))
        
        for i in range(len(sc1)):
            for j in range(len(sc2)):
                # 卡方距離
                chi2_dist = 0.5 * np.sum((sc1[i] - sc2[j])**2 / (sc1[i] + sc2[j] + 1e-10))
                cost_matrix[i, j] = chi2_dist
        
        return cost_matrix
    
    def thin_plate_spline_transform(self, img, source_points, target_points):
        """薄板樣條變換實現非剛性變形 - 改進版避免空洞"""
        # 確保數據類型一致
        source_points = np.array(source_points, dtype=np.float64)
        target_points = np.array(target_points, dtype=np.float64)
        
        if len(source_points) != len(target_points):
            min_len = min(len(source_points), len(target_points))
            source_points = source_points[:min_len]
            target_points = target_points[:min_len]
        
        h, w = img.shape
        n = len(source_points)
        
        if n < 3:
            # 如果點數太少，使用簡單的仿射變換
            return self.simple_affine_transform(img, source_points, target_points)
        
        # 使用改進的變換方法
        return self.improved_transform(img, source_points, target_points)
    
    def simple_affine_transform(self, img, source_points, target_points):
        """簡單仿射變換作為備用方法"""
        h, w = img.shape
        
        if len(source_points) >= 3 and len(target_points) >= 3:
            # 使用前三個點進行仿射變換
            src_pts = source_points[:3].astype(np.float32)
            dst_pts = target_points[:3].astype(np.float32)
            
            M = cv2.getAffineTransform(src_pts, dst_pts)
            result = cv2.warpAffine(img, M, (w, h))
            return result
        else:
            # 如果點數不足，返回原圖
            return img.copy()
    
    def improved_transform(self, img, source_points, target_points):
        """改進的圖像變換方法 - 使用反向映射避免空洞"""
        h, w = img.shape
        result = np.zeros_like(img)
        
        # 使用反向映射 - 對每個輸出像素找到對應的輸入像素
        for y in range(h):
            for x in range(w):
                # 當前輸出像素位置
                output_point = np.array([x, y], dtype=np.float64)
                
                # 找到最近的控制點來計算逆變換
                input_point = self.inverse_transform_point(output_point, source_points, target_points)
                
                # 雙線性插值獲取像素值
                pixel_value = self.bilinear_interpolate(img, input_point[0], input_point[1])
                result[y, x] = pixel_value
        
        return result
    
    def inverse_transform_point(self, output_point, source_points, target_points):
        """計算逆變換 - 從輸出點找到對應的輸入點"""
        if len(source_points) < 3:
            return output_point
        
        # 使用最近鄰方法進行近似逆變換
        # 找到最近的目標點
        distances = np.linalg.norm(target_points - output_point, axis=1)
        nearest_idx = np.argmin(distances)
        
        # 使用對應的源點作為基礎
        if distances[nearest_idx] < 1e-10:
            return source_points[nearest_idx]
        
        # 使用加權平均進行插值
        weights = 1.0 / (distances + 1e-10)
        weights = weights / np.sum(weights)
        
        input_point = np.sum(source_points * weights.reshape(-1, 1), axis=0)
        return input_point
    
    def bilinear_interpolate(self, img, x, y):
        """雙線性插值"""
        h, w = img.shape
        
        # 邊界檢查
        x = np.clip(x, 0, w-1)
        y = np.clip(y, 0, h-1)
        
        x1, y1 = int(x), int(y)
        x2, y2 = min(x1 + 1, w-1), min(y1 + 1, h-1)
        
        # 計算權重
        wx = x - x1
        wy = y - y1
        
        # 雙線性插值
        value = (img[y1, x1] * (1-wx) * (1-wy) +
                img[y1, x2] * wx * (1-wy) +
                img[y2, x1] * (1-wx) * wy +
                img[y2, x2] * wx * wy)
        
        return int(value)
    
    def apply_tps_transform(self, points, control_points, w):
        """應用TPS變換到點集"""
        transformed = np.zeros_like(points)
        
        for i, point in enumerate(points):
            # 仿射部分
            transformed[i] = point  # 初始為原始點
            
            # 非線性部分
            for j, cp in enumerate(control_points):
                r = np.linalg.norm(point - cp)
                if r > 0:
                    rbf_val = r**2 * np.log(r)
                    transformed[i] += w[j] * rbf_val
        
        return transformed
    
    def align_images(self, target_path, reference_path, output_dir='output'):
        """主要的圖片對齊函數"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 載入圖片
        target_img, target_gray = self.load_and_preprocess(target_path)
        ref_img, ref_gray = self.load_and_preprocess(reference_path)
        
        results = {}
        
        # 執行四種對齊方法
        for method_name, method_func in self.methods.items():
            try:
                print(f"\n正在執行 {method_name} 方法...")
                aligned_img = method_func(target_gray, ref_gray)
                
                # 如果是灰度圖，轉換為彩色以便保存
                if len(aligned_img.shape) == 2:
                    aligned_img = cv2.cvtColor(aligned_img, cv2.COLOR_GRAY2BGR)
                
                results[method_name] = aligned_img
                
                # 保存結果
                output_path = os.path.join(output_dir, f'{method_name}_aligned.jpg')
                cv2.imwrite(output_path, aligned_img)
                print(f"{method_name} 完成，結果保存至: {output_path}")
                
            except Exception as e:
                print(f"{method_name} 方法執行失敗: {str(e)}")
                results[method_name] = None
        
        # 創建比較圖
        self.create_comparison_plot(target_img, ref_img, results, output_dir)
        
        return results
    
    def create_comparison_plot(self, target_img, ref_img, results, output_dir):
        """創建比較圖表"""
        plt.figure(figsize=(15, 12))
        
        # 原始目標圖
        plt.subplot(3, 3, 1)
        plt.imshow(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
        plt.title('目標圖片 (Target)')
        plt.axis('off')
        
        # 參考圖
        plt.subplot(3, 3, 2)
        plt.imshow(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
        plt.title('參考圖片 (Reference)')
        plt.axis('off')
        
        # 四種方法的結果
        methods = ['RANSAC_Affine', 'EMD', 'CPD', 'Shape_Context']
        positions = [4, 5, 7, 8]
        
        for i, method in enumerate(methods):
            plt.subplot(3, 3, positions[i])
            if results[method] is not None:
                plt.imshow(cv2.cvtColor(results[method], cv2.COLOR_BGR2RGB))
                plt.title(f'{method} 對齊結果')
            else:
                plt.text(0.5, 0.5, f'{method}\n執行失敗', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title(f'{method} (失敗)')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'alignment_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()

# 使用範例
if __name__ == "__main__":
    # 創建對齊器實例
    aligner = ImageAlignment()
    
    # 執行圖片對齊
    # 請將這些路徑替換為您的實際圖片路徑
    target_image_path = "Strike_Analyzing/target.png"     # 目標圖片路徑
    reference_image_path = "Strike_Analyzing/test1.png"  # 參考圖片路徑
    
    try:
        results = aligner.align_images(target_image_path, reference_image_path)
        
        print("\n=== 對齊結果總結 ===")
        for method, result in results.items():
            status = "成功" if result is not None else "失敗"
            print(f"{method}: {status}")
            
    except FileNotFoundError as e:
        print(f"錯誤: 找不到圖片文件 - {e}")
        print("請確保圖片路徑正確，並將 target_image_path 和 reference_image_path 替換為實際的圖片路徑")
    except Exception as e:
        print(f"執行過程中發生錯誤: {e}")

# 說明：
print("""
使用說明：
1. 將 target_image_path 和 reference_image_path 替換為您的實際圖片路徑
2. 運行程式後會在 'output' 資料夾中生成：
   - RANSAC_Affine_aligned.jpg: RANSAC + 仿射變換結果
   - EMD_aligned.jpg: EMD對齊結果  
   - CPD_aligned.jpg: CPD對齊結果
   - Shape_Context_aligned.jpg: 形狀上下文匹配結果
   - alignment_comparison.png: 所有結果的比較圖

四種方法特點：
1. RANSAC + Affine: 基於特徵點匹配，處理不同數量的特徵點
2. EMD: 基於最優傳輸理論的點集對齊
3. CPD: 概率性點集配準，處理噪聲和離群點
4. Shape Context: 基於形狀描述子的匹配方法

所有方法都實現了非剛性變換，使用薄板樣條(TPS)進行最終的圖像變形。
""")

"""
import cv2
import numpy as np
from scipy.spatial import procrustes
from scipy.optimize import linear_sum_assignment
from pyemd import emd
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#print(os.getcwd())
from utils import *
# 座標格式轉換器
# 將多層嵌套的座標列表轉換為簡單的 [(),()]格式

def convert_coordinates(nested_coords):

    result = []
    
    def extract_coords(coords):

        if isinstance(coords, list):
            if len(coords) == 2 and all(isinstance(x, (int, float)) for x in coords):
                # 這是一個座標點 [x, y]
                return [tuple(coords)]
            else:
                # 這是一個包含多個元素的列表，遞歸處理
                extracted = []
                for item in coords:
                    extracted.extend(extract_coords(item))
                return extracted
        elif isinstance(coords, tuple) and len(coords) == 2:
            # 這已經是一個座標元組 (x, y)
            return [coords]
        else:
            return []
    
    # 對輸入的每個多邊形進行處理
    for polygon in nested_coords:
        polygon_coords = extract_coords(polygon)
        result.extend(polygon_coords)
    
    return result
# ========== Utility Functions ==========

def load_and_preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea).squeeze(axis=1)  # Return the largest contour

def save_transformation_matrix(matrix, filename):
    np.save(filename, matrix)


# ========== Alignment Methods ==========

def ransac_affine_transform(src, dst):
    matrix, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)
    return matrix


def procrustes_analysis(src, dst):
    mtx1, mtx2, _ = procrustes(src, dst)
    return mtx2  # Transformed dst points aligned to src


def emd_alignment(src, dst):
    def pairwise_dist(p1, p2):
        return np.linalg.norm(p1[:, None] - p2[None, :], axis=2).astype(np.float64)

    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    dist_matrix = pairwise_dist(src, dst)

    # Uniform weights
    src_weight = np.ones(len(src)) / len(src)
    dst_weight = np.ones(len(dst)) / len(dst)

    flow = emd(src_weight.tolist(), dst_weight.tolist(), dist_matrix.tolist())
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    aligned_dst = dst[col_ind]
    return aligned_dst


# ========== Main Comparison Pipeline ==========



# 主程式
if __name__ == "__main__":
    # --- 可調參數 ---
    img1_path = "Strike_Analyzing/target.png"
    img2_path = "Strike_Analyzing/test1.png"
    scale_factor = 2             # 前處理放大倍數
    final_shrink_factor = 0.5    # 縮小倍數
    blur_ksize = 3               # 模糊核大小  
    threshold_value = 180        # 二質化閾值
    epsilon = 1.0                # 簡化輪廓的誤差
    rdp_epsilon = 3              # RDP簡化閾值
    curvature_threshold = 30     # 曲率閾值
    min_radius = 10              # 最小搜尋半徑
    max_radius = 50              # 最大搜尋半徑
    debug = True                 # 是否打印除錯信息
    ifshow = 0                   # 是否中途顯示
    # ----------------

    try:
        contour1 = load_and_preprocess_image(img1_path)
        contour2 = load_and_preprocess_image(img2_path)

        # 原圖 灰階圖
        original_img1, gray_img1 = inputimg_colortogray(img1_path)
        original_img2, gray_img2 = inputimg_colortogray(img2_path)

        # 前處理圖片
        preprocessed_img1 = preprocess_image(gray_img1, scale_factor, blur_ksize, threshold_value, ifshow)
        preprocessed_img2 = preprocess_image(gray_img2, scale_factor, blur_ksize, threshold_value, ifshow)

        # 得到圖片輪廓
        contours1 = getContours(preprocessed_img1, ifshow)
        contours2 = getContours(preprocessed_img2, ifshow)

        svcfplist1 = []
        svcfplist2 = []
        # 處理每個輪廓
        for contour in contours1:
            
            fixcontour = [sublist[0] for sublist in contour]
            fixcontour = remove_consecutive_duplicates(fixcontour)  # 移除首尾或相鄰重複點
            rdp_points = rdp(fixcontour, epsilon=rdp_epsilon)
            print("RDP簡化後的點數:", len(rdp_points))

            custom_points, custom_idx = svcfp_queue(
                fixcontour,
                rdp_points,
                min_radius=min_radius,
                max_radius=max_radius,
                curvature_threshold=curvature_threshold,
                debug=debug,
                ifserver=0
            )
            svcfplist1.append(custom_points)
        for contour in contours2:
            
            fixcontour = [sublist[0] for sublist in contour]
            fixcontour = remove_consecutive_duplicates(fixcontour)  # 移除首尾或相鄰重複點
            rdp_points = rdp(fixcontour, epsilon=rdp_epsilon)
            print("RDP簡化後的點數:", len(rdp_points))

            custom_points, custom_idx = svcfp_queue(
                fixcontour,
                rdp_points,
                min_radius=min_radius,
                max_radius=max_radius,
                curvature_threshold=curvature_threshold,
                debug=debug,
                ifserver=0
            )
            svcfplist2.append(custom_points)


        keypoints1 = np.array(convert_coordinates(svcfplist1))
        keypoints2 = np.array(convert_coordinates(svcfplist2))


        # -- RANSAC + Affine --
        affine_matrix = ransac_affine_transform(keypoints2, keypoints1)
        
        
        if affine_matrix is not None:
            aligned_ransac = cv2.transform(np.array([keypoints2]), affine_matrix)[0]
            save_transformation_matrix(affine_matrix, "ransac_affine.npy")
        else:
            aligned_ransac = keypoints2

        # -- Procrustes --
        aligned_procrustes = procrustes_analysis(keypoints1, keypoints2)
        save_transformation_matrix(aligned_procrustes, "procrustes.npy")

        # -- EMD --
        aligned_emd = emd_alignment(keypoints1, keypoints2)
        save_transformation_matrix(aligned_emd, "emd_aligned.npy")

        # ========== Example Usage ==========




        aligned_ransac[:5], aligned_procrustes[:5], aligned_emd[:5], keypoints1[:5]

        
    except Exception as e:
        print(f"發生錯誤: {e}")
        import traceback
        traceback.print_exc()
"""