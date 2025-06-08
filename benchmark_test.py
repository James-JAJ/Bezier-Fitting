import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from typing import Dict, List, Tuple
from scipy.spatial import cKDTree
from skimage.metrics import structural_similarity as ssim
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
sys.stdout.reconfigure(encoding='utf-8')

class BMNDExperimentDesign:
    """設計專門凸顯BMND精確度的實驗"""
    
    def __init__(self, font_path=None):
        self.font_path = font_path
        self.results = {}
        # 定義一致的顏色方案 - 每個指標固定顏色
        self.color_scheme = {
            'BMND': '#FF6B6B',      # 紅色 - 突出我們的方法
            'MSE': '#4ECDC4',      # 青色
            'RMSE': '#45B7D1',     # 藍色
            'PSNR': '#96CEB4',     # 綠色
            'SSIM': '#FFEAA7'      # 黃色
        }
        
        # 設置字體
        self.setup_fonts()
    
    def setup_fonts(self):
        """設置字體，處理中文顯示問題"""
        try:
            if self.font_path and os.path.exists(self.font_path):
                self.font_prop = font_manager.FontProperties(fname=self.font_path)
                matplotlib.rcParams['font.sans-serif'] = [self.font_prop.get_name()] + matplotlib.rcParams['font.sans-serif']
                print(f"成功載入字體: {self.font_path}")
            else:
                # 嘗試使用系統中文字體
                possible_fonts = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'PingFang SC', 'Noto Sans CJK SC', 'DejaVu Sans']
                self.font_prop = None
                for font in possible_fonts:
                    try:
                        font_files = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
                        font_list = [matplotlib.font_manager.FontProperties(fname=fname).get_name() for fname in font_files]
                        if font in font_list or font in ['DejaVu Sans']:
                            matplotlib.rcParams['font.sans-serif'] = [font] + matplotlib.rcParams['font.sans-serif']
                            self.font_prop = font_manager.FontProperties(family=font)
                            print(f"使用系統字體: {font}")
                            break
                    except:
                        continue
                
                if self.font_prop is None:
                    print("警告: 無法載入中文字體，將使用英文標籤")
                    self.font_prop = font_manager.FontProperties(family='DejaVu Sans')
                    # 設置英文標籤
                    self.use_english = True
        except Exception as e:
            print(f"字體設置錯誤: {e}")
            self.font_prop = font_manager.FontProperties(family='DejaVu Sans')
            self.use_english = True
    
    def get_labels(self):
        """根據字體設置返回標籤"""
        if hasattr(self, 'use_english') and self.use_english:
            return {
                'shape_sensitivity': 'Shape Sensitivity Test',
                'progressive_distortion': 'Progressive Distortion Response',
                'outlier_robustness': 'Outlier Robustness Test',
                'test_case': 'Test Cases',
                'similarity_score': 'Similarity Score',
                'distortion_level': 'Distortion Level',
                'outlier_count': 'Number of Outliers',
                'rotation': 'Rotation',
                'summary': 'BMND Advantages Summary'
            }
        else:
            return {
                'shape_sensitivity': '實驗1: 形狀敏感度比較',
                'progressive_distortion': '實驗2: 漸進式扭曲響應',
                'outlier_robustness': '實驗3: 異常值穩健性',
                'test_case': '測試案例',
                'similarity_score': '相似度分數',
                'distortion_level': '扭曲程度',
                'outlier_count': '異常點數量',
                'rotation': '旋轉',
                'summary': 'BMND優勢總結'
            }
    
    def create_synthetic_shapes(self) -> Dict[str, np.ndarray]:
        """創建合成形狀用於控制實驗"""
        shapes = {}
        
        # 1. 基本圓形
        img = np.ones((200, 200), dtype=np.uint8) * 255
        cv2.circle(img, (100, 100), 50, 0, -1)
        shapes['circle_base'] = img
        
        # 2. 稍微變形的圓形（橢圓）
        img = np.ones((200, 200), dtype=np.uint8) * 255
        cv2.ellipse(img, (100, 100), (50, 45), 0, 0, 360, 0, -1)
        shapes['circle_slight_deform'] = img
        
        # 3. 明顯變形的圓形
        img = np.ones((200, 200), dtype=np.uint8) * 255
        cv2.ellipse(img, (100, 100), (50, 30), 0, 0, 360, 0, -1)
        shapes['circle_major_deform'] = img
        
        # 4. 相同面積但不同形狀
        img = np.ones((200, 200), dtype=np.uint8) * 255
        pts = np.array([[75,75], [125,75], [125,125], [75,125]], np.int32)
        cv2.fillPoly(img, [pts], 0)
        shapes['square_same_area'] = img
        
        # 5. 局部缺失的圓形
        img = np.ones((200, 200), dtype=np.uint8) * 255
        cv2.circle(img, (100, 100), 50, 0, -1)
        cv2.rectangle(img, (120, 90), (150, 110), 255, -1)  # 移除一小塊
        shapes['circle_local_missing'] = img
        
        # 6. 噪點干擾的圓形
        img = np.ones((200, 200), dtype=np.uint8) * 255
        cv2.circle(img, (100, 100), 50, 0, -1)
        # 添加隨機噪點
        np.random.seed(42)
        noise_points = np.random.randint(0, 200, (50, 2))
        for pt in noise_points:
            cv2.circle(img, tuple(pt), 2, 0, -1)
        shapes['circle_with_noise'] = img
        
        return shapes
    
    def BMND_shape_similarity(self, A, B):
        """你的BMND計算函數"""
        def contours_to_points(contours):
            if isinstance(contours, np.ndarray) and contours.ndim == 2 and contours.shape[1] == 2:
                return contours
            if isinstance(contours, list) and len(contours) > 0:
                if contours is None or len(contours) == 0:
                    return np.zeros((1, 2))
                valid = [c.reshape(-1, 2) for c in contours if c.shape[0] >= 5]
                return np.concatenate(valid, axis=0) if valid else np.zeros((1, 2))
            if isinstance(contours, np.ndarray):
                return contours.reshape(-1, 2)
            return np.zeros((1, 2))
        
        def mean_min_distance(X, Y):
            tree = cKDTree(X)
            dists, _ = tree.query(Y)
            return np.mean(dists)
        
        points_A = contours_to_points(A)
        points_B = contours_to_points(B)
        
        if len(points_A) < 2 or len(points_B) < 2:
            return 0.0
        
        avg_dist = (mean_min_distance(points_A, points_B) + mean_min_distance(points_B, points_A)) / 2
        sim = 1 / (1 + avg_dist)
        return (sim*100)
    
    def get_comparison_metrics(self):
        """其他比較指標"""
        return {
            'MSE': lambda img1, img2: 1 / (1 + np.mean((img1.astype("float") - img2.astype("float")) ** 2)),
            'RMSE': lambda img1, img2: 1 / (1 + np.sqrt(np.mean((img1.astype("float") - img2.astype("float")) ** 2))),
            'PSNR': self.psnr_similarity,
            'SSIM': lambda img1, img2: ssim(img1, img2),
        }
    
    def psnr_similarity(self, img1, img2):
        mse = np.mean((img1.astype("float") - img2.astype("float")) ** 2)
        if mse == 0:
            return 1.0
        PIXEL_MAX = 255.0
        psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
        return min(1.0, psnr / 100)
    
    def experiment_1_shape_sensitivity(self, shapes: Dict[str, np.ndarray]) -> Dict:
        """實驗1: 形狀敏感度測試"""
        print("=== 實驗1: 形狀敏感度測試 ===")
        
        base_shape = shapes['circle_base']
        _, bin_base = cv2.threshold(base_shape, 200, 255, cv2.THRESH_BINARY)
        A = np.argwhere(bin_base == 0)
        
        test_cases = [
            ('Slight Deform', 'circle_slight_deform'),
            ('Major Deform', 'circle_major_deform'),
            ('Same Area Diff Shape', 'square_same_area'),
            ('Local Missing', 'circle_local_missing'),
            ('With Noise', 'circle_with_noise')
        ]
        
        metrics = self.get_comparison_metrics()
        metrics['BMND'] = lambda img1, img2: 0  # 會被替換
        
        results = {}
        for name, shape_key in test_cases:
            test_shape = shapes[shape_key]
            _, bin_test = cv2.threshold(test_shape, 200, 255, cv2.THRESH_BINARY)
            B = np.argwhere(bin_test == 0)
            
            scores = {}
            for metric_name, metric_func in metrics.items():
                if metric_name == 'BMND':
                    scores[metric_name] = self.BMND_shape_similarity(A, B) / 100
                else:
                    scores[metric_name] = metric_func(base_shape, test_shape)
            
            results[name] = scores
            print(f"{name}:")
            for metric_name, score in scores.items():
                print(f"  {metric_name}: {score:.4f}")
        
        return results
    
    def experiment_2_progressive_distortion(self, base_img: np.ndarray) -> Dict:
        """實驗2: 漸進式扭曲測試"""
        print("\n=== 實驗2: 漸進式扭曲測試 ===")
        
        _, bin_base = cv2.threshold(base_img, 200, 255, cv2.THRESH_BINARY)
        A = np.argwhere(bin_base == 0)
        
        # 不同類型的扭曲
        distortion_types = {
            'Rotation': lambda img, level: self.apply_rotation(img, level * 45),
            'Scaling': lambda img, level: self.apply_scaling(img, 1 + level * 0.5),
            'Translation': lambda img, level: self.apply_translation(img, level * 20),
            'Blur': lambda img, level: self.apply_blur(img, level * 3),
            'Noise': lambda img, level: self.apply_noise(img, level * 50)
        }
        
        levels = np.linspace(0, 1, 11)
        metrics = self.get_comparison_metrics()
        
        results = {}
        for dist_name, dist_func in distortion_types.items():
            print(f"測試 {dist_name} 扭曲...")
            dist_results = {metric: [] for metric in ['BMND'] + list(metrics.keys())}
            
            for level in levels:
                distorted = dist_func(base_img.copy(), level)
                _, bin_dist = cv2.threshold(distorted, 200, 255, cv2.THRESH_BINARY)
                B = np.argwhere(bin_dist == 0)
                
                # BMND分數
                BMND_score = self.BMND_shape_similarity(A, B) / 100
                dist_results['BMND'].append(BMND_score)
                
                # 其他指標分數
                for metric_name, metric_func in metrics.items():
                    try:
                        score = metric_func(base_img, distorted)
                        dist_results[metric_name].append(score)
                    except:
                        dist_results[metric_name].append(0.0)
            
            results[dist_name] = dist_results
        
        return results
    
    def apply_rotation(self, img, angle):
        rows, cols = img.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        return cv2.warpAffine(img, M, (cols, rows), borderValue=255)
    
    def apply_scaling(self, img, scale):
        rows, cols = img.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 0, scale)
        return cv2.warpAffine(img, M, (cols, rows), borderValue=255)
    
    def apply_translation(self, img, tx):
        rows, cols = img.shape
        M = np.float32([[1, 0, tx], [0, 1, tx]])
        return cv2.warpAffine(img, M, (cols, rows), borderValue=255)
    
    def apply_blur(self, img, sigma):
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(img, sigma=sigma)
    
    def apply_noise(self, img, noise_level):
        noise = np.random.normal(0, noise_level, img.shape)
        noisy = img.astype(float) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def experiment_3_outlier_robustness(self, base_img: np.ndarray) -> Dict:
        """實驗3: 異常值穩健性測試"""
        print("\n=== 實驗3: 異常值穩健性測試 ===")
        
        _, bin_base = cv2.threshold(base_img, 200, 255, cv2.THRESH_BINARY)
        A = np.argwhere(bin_base == 0)
        
        # 創建不同程度的異常值
        outlier_tests = []
        for i in range(1, 6):
            # 在基礎形狀上添加隨機點
            outlier_img = base_img.copy()
            np.random.seed(42 + i)
            n_outliers = i * 10
            outlier_points = np.random.randint(20, 180, (n_outliers, 2))
            for pt in outlier_points:
                cv2.circle(outlier_img, tuple(pt), 2, 0, -1)
            
            outlier_tests.append((f'{n_outliers} Outliers', outlier_img))
        
        metrics = self.get_comparison_metrics()
        results = {}
        
        for test_name, test_img in outlier_tests:
            _, bin_test = cv2.threshold(test_img, 200, 255, cv2.THRESH_BINARY)
            B = np.argwhere(bin_test == 0)
            
            scores = {}
            scores['BMND'] = self.BMND_shape_similarity(A, B) / 100
            
            for metric_name, metric_func in metrics.items():
                try:
                    scores[metric_name] = metric_func(base_img, test_img)
                except:
                    scores[metric_name] = 0.0
            
            results[test_name] = scores
            print(f"{test_name}:")
            for metric_name, score in scores.items():
                print(f"  {metric_name}: {score:.4f}")
        
        return results
    
    def visualize_experiment_1(self, exp1_results):
        """實驗1視覺化 - 形狀敏感度"""
        labels = self.get_labels()
        
        plt.figure(figsize=(12, 8))
        metrics = list(next(iter(exp1_results.values())).keys())
        test_cases = list(exp1_results.keys())
        
        x = np.arange(len(test_cases))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            scores = [exp1_results[case][metric] for case in test_cases]
            color = self.color_scheme.get(metric, f'C{i}')
            plt.bar(x + i*width, scores, width, label=metric, alpha=0.8, color=color)
        
        plt.xlabel(labels['test_case'], fontproperties=self.font_prop, fontsize=12)
        plt.ylabel(labels['similarity_score'], fontproperties=self.font_prop, fontsize=12)
        plt.title(labels['shape_sensitivity'], fontproperties=self.font_prop, fontsize=14, fontweight='bold')
        plt.xticks(x + width * 2, test_cases, rotation=45, ha='right')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def visualize_experiment_2(self, exp2_results):
        """實驗2視覺化 - 漸進式扭曲"""
        labels = self.get_labels()
        
        plt.figure(figsize=(12, 8))
        distortion_type = 'Rotation'  # 選擇旋轉作為展示
        
        if distortion_type in exp2_results:
            levels = np.linspace(0, 1, 11)
            for metric, scores in exp2_results[distortion_type].items():
                color = self.color_scheme.get(metric, None)
                plt.plot(levels, scores, 'o-', label=metric, alpha=0.8, linewidth=2, 
                        markersize=6, color=color)
        
        plt.xlabel(labels['distortion_level'], fontproperties=self.font_prop, fontsize=12)
        plt.ylabel(labels['similarity_score'], fontproperties=self.font_prop, fontsize=12)
        plt.title(f"{labels['progressive_distortion']} - {labels['rotation']}", 
                 fontproperties=self.font_prop, fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()
    
    def visualize_experiment_3(self, exp3_results):
        """實驗3視覺化 - 異常值穩健性"""
        labels = self.get_labels()
        
        plt.figure(figsize=(12, 8))
        test_cases = list(exp3_results.keys())
        metrics = list(next(iter(exp3_results.values())).keys())
        
        x = np.arange(len(test_cases))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            scores = [exp3_results[case][metric] for case in test_cases]
            color = self.color_scheme.get(metric, f'C{i}')
            plt.bar(x + i*width, scores, width, label=metric, alpha=0.8, color=color)
        
        plt.xlabel(labels['outlier_count'], fontproperties=self.font_prop, fontsize=12)
        plt.ylabel(labels['similarity_score'], fontproperties=self.font_prop, fontsize=12)
        plt.title(labels['outlier_robustness'], fontproperties=self.font_prop, fontsize=14, fontweight='bold')
        plt.xticks(x + width * 2, test_cases, rotation=45, ha='right')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def run_all_experiments(self):
        """執行所有實驗"""
        print("開始BMND精確度評估實驗...")
        
        # 創建測試形狀
        shapes = self.create_synthetic_shapes()
        
        # 執行實驗
        exp1_results = self.experiment_1_shape_sensitivity(shapes)
        exp2_results = self.experiment_2_progressive_distortion(shapes['circle_base'])
        exp3_results = self.experiment_3_outlier_robustness(shapes['circle_base'])
        
        # 分別視覺化三個實驗（三張圖）
        print("\n顯示實驗1結果...")
        self.visualize_experiment_1(exp1_results)
        
        print("\n顯示實驗2結果...")
        self.visualize_experiment_2(exp2_results)
        
        print("\n顯示實驗3結果...")
        self.visualize_experiment_3(exp3_results)
        
        return {
            'shape_sensitivity': exp1_results,
            'progressive_distortion': exp2_results,
            'outlier_robustness': exp3_results
        }

# 使用範例
if __name__ == '__main__':
    font_path = "benchmarkimg/NotoSansTC-VariableFont_wght.ttf"  # 根據你的路徑調整
    
    experiment = BMNDExperimentDesign(font_path)
    results = experiment.run_all_experiments()
