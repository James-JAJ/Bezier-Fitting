import numpy  as np
import random
from scipy.spatial import cKDTree
import sys
import os
sys.path.append(os.path.abspath("/utils"))
from .bezier import *
from .math_tools import *
from .server_tools import *


#genetic algorithm
#genetic_algorithm:使用遺傳演算法進行擬合
def genetic_algorithm(target_curve, p1, p4, width, height, pop_size=50, generations=500):

    
    # 豪斯多夫距離計算
    def hausdorff_distance(set1, set2):
        """計算兩個點集之間的 Hausdorff 距離。"""
        tree1 = cKDTree(set1)
        tree2 = cKDTree(set2)
        dist1, _ = tree1.query(set2)
        dist2, _ = tree2.query(set1)
        return max(np.max(dist1), np.max(dist2))

    def average_distance(set1, set2):
        """計算兩個點集之間的平均距離。"""
        tree1 = cKDTree(set1)
        tree2 = cKDTree(set2)
        dist1, _ = tree1.query(set2)
        dist2, _ = tree2.query(set1)
        return (np.mean(dist1) + np.mean(dist2)) / 2

    def fitness(individual, target, width, height, alpha=0.9, beta=0.1):
        """適應度函數，結合 Hausdorff 距離和平均距離。"""
        p2 = (int(individual[0]), int(individual[1]))
        p3 = (int(individual[2]), int(individual[3]))
        candidate = [p1, p2, p3, p4]
        candidate_curve = bezier_curve_calculate(candidate)

        max_possible_dist = (width**2 + height**2)**0.5
        """
        if len(candidate_curve)-len(remove_duplicates(candidate_curve))>0:
            return 50
        """
        
        hd = hausdorff_distance(candidate_curve, target)
        ad = average_distance(candidate_curve, target)
        se = len(target)-find_common_elements(candidate_curve, target)
        # 結合 Hausdorff 距離和平均距離
        combined_distance = alpha * hd + beta * ad 
        
        #使用不同的標準化參數
        normalized = np.exp(-2 * combined_distance / max_possible_dist)
        return normalized * 100
    # 種群初始化 (允許在更大範圍內隨機生成控制點)
    def initialize_population(size):
        population = []
        for _ in range(size):
            # 在p1和p4之間隨機初始化控制點
            x2 = random.uniform(min(p1[0], p4[0]), max(p1[0], p4[0]))
            y2 = random.uniform(min(p1[1], p4[1]), max(p1[1], p4[1]))
            x3 = random.uniform(min(p1[0], p4[0]), max(p1[0], p4[0]))
            y3 = random.uniform(min(p1[1], p4[1]), max(p1[1], p4[1]))
            population.append([x2, y2, x3, y3])
        return population    
    # 改良版選擇函數
    def selection(population, scores):
        # 處理全零適應度情況
        if sum(scores) <= 0:
            return [random.choice(population), random.choice(population)]
        
        # 轉換為機率分布
        probabilities = np.array(scores)/sum(scores)
        
        # 使用numpy隨機選擇
        indices = np.random.choice(
            len(population),
            size=2,
            p=probabilities,
            replace=False
        )
        return [population[i] for i in indices]
    
    # 增強型交叉操作
    def crossover(parent1, parent2, crossover_rate=0.75):
        if random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # 將列表轉換為NumPy陣列
        p1 = np.array(parent1)
        p2 = np.array(parent2)
        
        # 多點交叉
        mask = np.random.randint(0, 2, size=len(parent1), dtype=bool)
        child1 = np.where(mask, p1, p2)
        child2 = np.where(mask, p2, p1)
        
        # 轉回列表
        return child1.tolist(), child2.tolist()
    
    # 智慧型變異操作 - 允許更大範圍的探索
    def mutate(individual, mutation_rate=0.5):
        result = individual.copy()

        for i in range(len(result)):
            if random.random() < mutation_rate:
                mutation_strength = 30 * (1 + random.random())
                result[i] += np.random.normal(0, mutation_strength)
        return result
    # 遺傳演算法主流程
    # 初始化種群
    population = initialize_population(pop_size)
    best_ever = None
    best_score = -np.inf
    last_score = 0
    consecutive_no_improvement = 0

    #target_curve=rdp(target_curve,0.001)
    
    print(len(target_curve))
    for gen in range(generations):
        # 計算適應度
        scores = [fitness(ind, target_curve, width, height) for ind in population]
        
        # 更新全局最佳
        current_best_idx = np.argmax(scores)
        if scores[current_best_idx] > best_score:
            best_score = scores[current_best_idx]
            best_ever = population[current_best_idx].copy()
        
        # 每10代輸出準確度
        if (gen+1) % 10 == 0:
            custom_print(f"Generation {gen+1}: Best Score = {best_score:.2f}")
            
            # Early stop邏輯 - 如同原本的實現
            if best_score - last_score < 0.01:
                consecutive_no_improvement += 1
            else:
                consecutive_no_improvement = 0
            
            last_score = best_score
            
            # 連續四次沒有明顯改善則提前結束
            if consecutive_no_improvement == 4:
                custom_print(f"Early stopping at generation {gen+1}: No significant improvement for 40 generations")
                break
        
        # 新一代種群
        new_pop = []
        if best_ever is not None:
            new_pop.append(best_ever.copy())  # 精英保留
        
        while len(new_pop) < pop_size:
            # 選擇父母
            parents = selection(population, scores)
            
            # 交叉產生後代
            child1, child2 = crossover(parents[0], parents[1])
            
            # 變異並加入新種群
            new_pop.append(mutate(child1))
            if len(new_pop) < pop_size:
                new_pop.append(mutate(child2))
        
        population = new_pop[:pop_size]
    
    if best_ever is None:
        # 如果沒有找到最佳解，提供一個合理的默認值
        location = [p1, p1, p4, p4]
    else:
        # 保留原始值，不限制控制點必須在畫布內
        location = [p1, 
                   (int(best_ever[0]), int(best_ever[1])), 
                   (int(best_ever[2]), int(best_ever[3])), 
                   p4]
    
    return location