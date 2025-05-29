import numpy  as np
import random
from scipy.spatial import cKDTree
import sys
import os
sys.path.append(os.path.abspath("/utils"))
from .bezier import *
from .math_tools import *
from .server_tools import *

# ===============================================================
# ✅ 遺傳演算法進行貝茲曲線擬合（控制點優化）
# ---------------------------------------------------------------
# 本演算法針對固定端點 P0, P3，自動尋找中間控制點 P1, P2，
# 以最小化候選貝茲曲線與目標曲線的差異為目標。
# 評分指標以 Hausdorff 距離為主，平均距離為輔。
# ===============================================================

def genetic_algorithm(target_curve, p1, p4, width, height, pop_size=50, generations=500, ifserver=1):
    """
    使用遺傳演算法進行內部雙控制點擬合

    Args:
        target_curve (list of tuple): 目標擬合曲線 Datatype: [(x1,y1), (x2,y2), ..., (xn,yn)]
        p1, p4 (tuple): 三次貝茲曲線之首尾座標點
        width, height (int): 擬合畫布最大長寬（影響標準化）
        pop_size (int): 種群數量
        generations (int): 最大迭代次數
        ifserver (int): 是否輸出至伺服器日誌

    Returns:
        location (list of tuple): 最佳貝茲控制點 Datatype: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]

    ⚠️ 備註:
        - 控制點中僅內部兩點會被演算法優化
        - 初始點集經隨機化產生，包含變異與交配
        - 若多代無進展會提前停止（early stop）
    """

    def hausdorff_distance(set1, set2):
        """最大對最小距離：常用於評估形狀吻合程度（不容許偏移）"""
        tree1 = cKDTree(set1)
        tree2 = cKDTree(set2)
        dist1, _ = tree1.query(set2)
        dist2, _ = tree2.query(set1)
        return max(np.max(dist1), np.max(dist2))

    def average_distance(set1, set2):
        """雙向平均最短距離：比 Hausdorff 溫和些"""
        tree1 = cKDTree(set1)
        tree2 = cKDTree(set2)
        dist1, _ = tree1.query(set2)
        dist2, _ = tree2.query(set1)
        return (np.mean(dist1) + np.mean(dist2)) / 2

    def fitness(individual, target, width, height, alpha=0.9, beta=0.1):
        """
        適應度函數，結合 Hausdorff 距離與平均距離（權重可調）
        分數經指數轉換後落在 0~100 範圍，越大越佳
        """
        p2 = (int(individual[0]), int(individual[1]))
        p3 = (int(individual[2]), int(individual[3]))
        candidate = [p1, p2, p3, p4]
        candidate_curve = bezier_curve_calculate(candidate)
        max_possible_dist = (width**2 + height**2)**0.5
        hd = hausdorff_distance(candidate_curve, target)
        ad = average_distance(candidate_curve, target)
        combined_distance = alpha * hd + beta * ad
        normalized = np.exp(-2 * combined_distance / max_possible_dist)
        return normalized * 100

    def initialize_population(size):
        """初始族群產生：控制點 P2, P3 在 P1~P4 區間隨機分布"""
        population = []
        for _ in range(size):
            x2 = random.uniform(min(p1[0], p4[0]), max(p1[0], p4[0]))
            y2 = random.uniform(min(p1[1], p4[1]), max(p1[1], p4[1]))
            x3 = random.uniform(min(p1[0], p4[0]), max(p1[0], p4[0]))
            y3 = random.uniform(min(p1[1], p4[1]), max(p1[1], p4[1]))
            population.append([x2, y2, x3, y3])
        return population

    def selection(population, scores):
        """機率選擇：分數越高，機率越高"""
        if sum(scores) <= 0:
            return [random.choice(population), random.choice(population)]
        probabilities = np.array(scores)/sum(scores)
        indices = np.random.choice(len(population), size=2, p=probabilities, replace=False)
        return [population[i] for i in indices]

    def crossover(parent1, parent2, crossover_rate=0.75):
        """隨機遮罩交配法，混合控制點座標"""
        if random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
        p1 = np.array(parent1)
        p2 = np.array(parent2)
        mask = np.random.randint(0, 2, size=len(parent1), dtype=bool)
        child1 = np.where(mask, p1, p2)
        child2 = np.where(mask, p2, p1)
        return child1.tolist(), child2.tolist()

    def mutate(individual, mutation_rate=0.5):
        """高斯擾動：對部分座標加入隨機變化"""
        result = individual.copy()
        for i in range(len(result)):
            if random.random() < mutation_rate:
                mutation_strength = 30 * (1 + random.random())
                result[i] += np.random.normal(0, mutation_strength)
        return result

    # ===== 遺傳主流程 =====
    population = initialize_population(pop_size)
    best_ever = None
    best_score = -np.inf
    last_score = 0
    consecutive_no_improvement = 0

    print(len(target_curve))
    for gen in range(generations):
        scores = [fitness(ind, target_curve, width, height) for ind in population]
        current_best_idx = np.argmax(scores)
        if scores[current_best_idx] > best_score:
            best_score = scores[current_best_idx]
            best_ever = population[current_best_idx].copy()

        if (gen+1) % 10 == 0:
            custom_print(ifserver, f"Generation {gen+1}: Best Score = {best_score:.2f}")
            if best_score - last_score < 0.01:
                consecutive_no_improvement += 1
            else:
                consecutive_no_improvement = 0
            last_score = best_score
            if consecutive_no_improvement == 4:
                custom_print(ifserver, f"Early stopping at generation {gen+1}: No significant improvement for 40 generations")
                break

        new_pop = []
        if best_ever is not None:
            new_pop.append(best_ever.copy())
        while len(new_pop) < pop_size:
            parents = selection(population, scores)
            child1, child2 = crossover(parents[0], parents[1])
            new_pop.append(mutate(child1))
            if len(new_pop) < pop_size:
                new_pop.append(mutate(child2))
        population = new_pop[:pop_size]

    if best_ever is None:
        location = [p1, p1, p4, p4]
    else:
        location = [p1, 
                    (int(best_ever[0]), int(best_ever[1])),
                    (int(best_ever[2]), int(best_ever[3])),
                    p4]
    return location
