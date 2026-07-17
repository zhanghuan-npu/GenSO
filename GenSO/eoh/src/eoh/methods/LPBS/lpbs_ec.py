import random

import hdbscan
import numpy as np
from sympy import false

from .lpbs_llm import Evolution
import warnings

class InterfaceEC:
    def __init__(self, pop_size, api_endpoint, api_key, llm_model, debug_mode, interface_prob, timeout):
        """
        初始化 InterfaceEC 类实例

        参数:
            pop_size (int)        : 种群大小
            m                     : 其他参数（如变异比例）
            api_endpoint (str)    : LLM API 接口地址
            api_key (str)         : API Key
            llm_model (str)       : 使用的 LLM 模型
            debug_mode (bool)     : 是否开启调试模式
            interface_prob        : 问题接口对象，用于计算适应度
            select                : 选择策略
            n_p                   : 其他参数
            timeout               : 超时时间
            use_numba (bool)      : 是否使用 numba 加速
            **kwargs              : 其他可选参数
        功能:
            - 初始化 LLM 接口实例
            - 保存算法运行参数和调试模式设置
            - 配置警告忽略策略（非调试模式下）
        """
        # LLM settings
        self.pop_size = pop_size
        self.interface_eval = interface_prob
        prompts = interface_prob.prompts
        self.llm = Evolution(api_endpoint, api_key, llm_model, debug_mode, prompts)
        self.debug = debug_mode

        if not self.debug:
            warnings.filterwarnings("ignore")

        self.timeout = timeout

    def init_population(self, pop_size):
        """
        初始化种群

        参数:
            pop_size (int): 要生成的个体数量

        返回: population (list[dict]): 初始化生成的个体列表，每个个体为字典，包含至少 'code' 字段

        功能:
            - 调用 self.llm.new_individual() 生成新的个体
            - 将生成的个体加入种群列表并返回
        """
        population = []

        while len(population) < pop_size:
            ind = self.llm.new_individual()
            repaired_ind = self.pilot_run_and_repair(ind)
            population.append(repaired_ind)
        return population

    def pilot_run_and_repair(self, individual):
        """
        对个体进行试运行 (pilot_run)，若失败则尝试修复 (repair)
        仅尝试 3 次修复，以免多次询问消耗时间

        输入:
            individual (dict): 初始个体，包含至少 'code' 字段

        返回:
            dict 修复后的个体
        """
        max_attempts = 2
        attempt = 0

        while attempt < max_attempts:
            # 尝试运行个体
            individual = self.interface_eval.pilot_run(individual)
            if individual.get("exec_success", True):
                return individual  # 成功，直接返回
            # 如果失败，尝试修复
            individual = self.llm.repair(individual)
            attempt += 1
        # 尝试次数用尽，返回最后一次修复后的个体
        return individual

    def evaluate_population(self, population: list[dict]) -> tuple[list[dict], int]:
        """
        评估种群中个体的适应度

        参数:
            population (list[dict]) : 个体列表，每个个体为字典，至少包含以下字段：
            - 'code' (str)          : LLM 生成的 Python 算法代码字符串
            - 'exec_success' (bool) : 标记代码是否成功执行过
            - 'case_fitness'        : 算法在每一个案例上的表现
            - 'objective' (float)   : 算法目标值（适应度），未评估时为 None
            - 'other_inf'           : 其他可选信息
            - 'traceback_msg' (str) : 执行失败时的错误信息
        返回值:
            tuple:
                - evaluated_population (list[dict]): 更新了评估结果的种群
                - eval_count (int): 本次调用实际评估的个体数量
        功能:
            - 遍历种群中的每个个体
            - 如果个体尚未评估（'obj' 为 None），调用 self.interface_eval.evaluate_ind(ind)
            - 统计实际评估的个体数量
        """
        eval_count = 0
        for ind in population:
            # 如果个体已经有目标值，则跳过评估
            if ind.get('objective') is None:
                self.interface_eval.evaluate_case_fitness(ind)
                eval_count += 1
        return population, eval_count


    def cluster_cases(self, parents: list[dict]) -> dict[int, list[int]]:
        """
        对案例在不同规则下的表现进行归一化和聚类

        参数：
            parents (list[dict]): 父代规则列表，每个元素包含 'case_fitness' 列表

        返回：
            dict[int, list[int]]: 聚类结果字典，键为聚类标签（int），值为属于该聚类的案例索引列表（list[int]）。
        """
        print("\n正在聚类案例")
        # 1. 构建父代规则案例表现矩阵 shape: (num_parents, num_cases)
        G = np.array([ind['case_fitness'] for ind in parents], dtype=float)
        # 2. 对每列（案例）进行归一化
        for c in range(G.shape[1]):
            col = G[:, c]
            min_val = col.min()
            max_val = col.max()
            if max_val > min_val:
                G[:, c] = (col - min_val) / (max_val - min_val)
            else:
                G[:, c] = 1  # 如果该列所有值相等，则设为 0
        # 3. 转置为 (num_cases, num_parents)
        X = G.T
        # 4. HDBSCAN 聚类案例
        # best_labels = None
        # min_total_clusters = np.inf
        # for min_size in range(2, 11):  # 尝试 min_cluster_size 从2到10
        #     clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size)
        #     labels = clusterer.fit_predict(X)
        #
        #     # 将噪声点（-1）重新标记为新聚类
        #     max_label = labels.max() if len(labels) > 0 else -1
        #     temp_labels = labels.copy()
        #     for i, lbl in enumerate(temp_labels):
        #         if lbl == -1:
        #             max_label += 1
        #             temp_labels[i] = max_label
        #
        #     total_clusters = len(np.unique(temp_labels))
        #     if total_clusters < min_total_clusters:
        #         min_total_clusters = total_clusters
        #         best_labels = temp_labels
        # labels = best_labels
        # print(f"最优聚类子类数量: {min_total_clusters}")
        # 使用总子类数最少的聚类结果
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
        labels = clusterer.fit_predict(X)

        # 将噪声点（-1）重新标记为新聚类
        max_label = labels.max() if len(labels) > 0 else -1
        temp_labels = labels.copy()
        for i, lbl in enumerate(temp_labels):
         if lbl == -1:
             max_label += 1
             temp_labels[i] = max_label
        print(f"聚类子类数量: {len(np.unique(temp_labels))}")
        # 6. 构建聚类字典：label -> 案例索引列表
        clusters = {}
        for case_idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(case_idx)
        return clusters

    def extract_modes(self, parents: list[dict], clusters: dict[int, list[int]]) -> list[str]:
        """
        为每个父代规则提取其最擅长的案例子类对应的模式。

        参数：
            parents (list[dict]): 父代规则列表，每个个体包含 'case_fitness'。
            clusters (dict[int, list[int]]): 案例子类字典，键为子类标签，值为案例索引列表。

        返回：
            list[str]: 模式字符串列表，每个元素是对应父母规则最擅长的子类模式。
        """
        print("\n正在提取模式")
        modes = []
        for parent_idx, parent in enumerate(parents):
            best_score = -np.inf
            best_cluster = None

            # 遍历所有子类，找到该父母最擅长的子类
            for cluster_label, case_list in clusters.items():
                avg_score = np.mean([parent['case_fitness'][c] for c in case_list])
                if avg_score > best_score:
                    best_score = avg_score
                    best_cluster = case_list

            # 调用 get_mode 生成模式字符串
            if best_cluster is not None:
                mode_str = self.llm.get_mode(best_cluster, parent)
            else:
                mode_str = "[Mode]: No suitable cluster found."
            modes.append(mode_str)
            # print(f"父母 {parent_idx + 1} 的模式:\n{mode_str}")
        return modes

    def crossover_population(self, parents, modes, pop_size):
        """
        对父代种群进行交叉生成后代。

        参数：
            parents (list[dict]): 父代个体列表
            modes (list[str]): 模式描述列表，用于指导交叉生成
            pop_size (int): 种群总规模

        返回：
            offsprings (list[dict]): 生成的后代个体列表
        """
        print("正在交叉种群")
        offsprings = []
        num_offsprings = int(pop_size * 0.5)
        for _ in range(num_offsprings):
            ind = self.llm.crossover(parents, modes)
            repaired_ind = self.pilot_run_and_repair(ind)
            offsprings.append(repaired_ind)
        offsprings, _ = self.evaluate_population(offsprings)
        # print("后代信息：")
        # for idx, ind in enumerate(offsprings, start=1):
            # print(f"Offspring {idx}: {ind.get('objective', None)}")
        return offsprings

    def mutate_population(self, population, clusters, modes, pop_size):
        """
        对种群进行基于模式的精英变异。

        参数：
            population (list[dict]): 当前种群个体列表，每个个体包含 'case_fitness' 列表
            clusters (dict): 聚类字典 {label: [case_indices]}
            pop_size (int): 种群总规模（目标，不是 population 长度）

        返回：
            offsprings (list[dict]): 变异生成的后代列表
        """
        print("正在变异种群")
        num_ind = len(population)
        num_mode = len(clusters)

        # 1. 构建个体 x 模式矩阵，保存每个个体在每个模式上的平均表现
        fitness_matrix = np.zeros((num_ind, num_mode))
        for i, ind in enumerate(population):
            for mode_idx, case_list in enumerate(clusters.values()):
                fitness_matrix[i, mode_idx] = np.mean([ind['case_fitness'][c] for c in case_list])
        # 2. 找出每个模式上最高的案例适应度
        best_mode_scores = np.max(fitness_matrix, axis=0)
        # 3 找到每个模式表现最好的个体索引
        best_ind_of_mode = np.argmax(fitness_matrix, axis=0)
        # 4. 找出每个个体表现最差的模式索引
        worst_mode_of_ind = np.argmin(fitness_matrix, axis=1)

        # 4. 对每个个体计算 替换最差模式后的总适应度
        # total_scores = np.zeros(num_ind)
        # for i in range(num_ind):
        #     total_score = 0
        #     for mode_idx, case_list in enumerate(clusters.values()):
        #         if mode_idx == worst_mode_of_ind[i]:
        #             # 用该模式上的最佳个体的表现替换
        #             total_score += np.sum([best_mode_scores[mode_idx] for _ in case_list])
        #         else: # 其他模式保持原有表现
        #             total_score += np.sum([fitness_matrix[i, mode_idx] for _ in case_list])
        #     total_scores[i] = total_score
        total_scores = np.sum(fitness_matrix, axis=1)
        # 5. 选择精英个体进行模式变异
        elite_num = int(pop_size * 0.5)
        # elite_ind = np.argsort(total_scores)[::-1][:elite_num]
        elite_ind = np.argsort(total_scores)[::-1][:elite_num]
        # 6. 对精英个体进行模式变异
        # 缓存已提取的模式
        mode_cache = {}
        # 后代列表
        offsprings = []
        for idx in elite_ind:
            # 待变异个体
            individual = population[idx]
            # 最差模式
            worst_mode = worst_mode_of_ind[idx]

            # 检查缓存
            if worst_mode in mode_cache:
                mode_description = mode_cache[worst_mode]
            else:
                # 否则提取新模式
                case_list = list(clusters.values())[worst_mode]
                best_individual = population[best_ind_of_mode[worst_mode]]
                mode_description = self.llm.get_mode(case_list, best_individual)
                mode_cache[worst_mode] = mode_description
                modes.append(mode_description)
            # print(f"\n[变异模式 - 模式 {worst_mode}]:\n{mode_description}")

            # 对精英个体进行变异
            mutated_offspring = self.llm.mutate(individual, mode_description)
            repaired_ind = self.pilot_run_and_repair(mutated_offspring)
            offsprings.append(repaired_ind)
        offsprings, _ = self.evaluate_population(offsprings)
        # print("后代信息：")
        # for idx, ind in enumerate(offsprings, start=1):
        #    print(f"Offspring {idx}: {ind.get('objective', None)}")
        return offsprings

    def population_management(self, population, offsprings):
        """
        种群管理函数：将后代逐个加入种群，并保持种群只保留 Pareto 非支配解集。

        参数：
            population (list[dict]): 当前种群列表，每个个体包含 'case_fitness' (list[float])
            offsprings (list[dict]): 新生成的后代个体列表

        返回：
            population (list[dict]): 更新后的非支配解种群
        """

        def dominates(a, b):
            """
            判断个体 a 是否支配个体 b。
            支配定义：a 在所有目标上都 >= b
            """
            """
            a_fit, b_fit = a['case_fitness'], b['case_fitness']
            for x, y in zip(a_fit, b_fit):
                if x < y:
                    return False
            return True
            """
            a_fit, b_fit = a['case_fitness'], b['case_fitness']

            better_or_equal_all = True
            strictly_better = False

            for x, y in zip(a_fit, b_fit):
                if x < y:
                    better_or_equal_all = False
                    break
                if x > y:
                    strictly_better = True

            return better_or_equal_all and strictly_better

        # 逐个将 offspring 加入 population
        for child in offsprings:
            population.append(child)

        # 种群去除支配解：只保留 Pareto 前沿
        non_dominated = []
        for i, ind in enumerate(population):
            dominated_flag = False
            for j, other in enumerate(population):
                if i != j and dominates(other, ind):
                    # ind 被 other 支配，舍弃
                    dominated_flag = True
                    break
            if not dominated_flag:
                non_dominated.append(ind)

        # 假如非支配集只剩1个个体
        if len(non_dominated) < 2:
            remaining = [ind for ind in population if ind not in non_dominated]
            if remaining:
                best = max(remaining, key=lambda x: x['objective'])
                non_dominated.append(best)

        # 更新 population 为当前非支配解集
        population = non_dominated

        return population

    def get_best_individual(self, population):
        """
        使用循环从种群中找到最好（objective 最大）的个体。

        参数：
            population (list[dict]): 种群列表，每个个体包含 'objective' (float/int)

        返回：
            best_ind (dict): objective 最优的个体
        """
        if not population:
            return None  # 空种群直接返回 None

        # 初始化，用第一个个体作为当前最优
        best_ind = population[0]

        # 遍历种群，逐个比较
        for ind in population[1:]:
            if ind['objective'] > best_ind['objective']:  # 如果当前个体更优
                best_ind = ind  # 更新最优个体

        return best_ind

    def diverse_initialization(self, pop_size):
        print("正在多样初始化")
        tasks = self.llm.get_task_description(pop_size)
        population = []
        for i in range(pop_size):
            if i < len(tasks):
                individual = self.llm.new_individual_of_task(tasks[i])
            else:
                individual = self.llm.new_individual()
            repaired_ind = self.pilot_run_and_repair(individual)
            population.append(repaired_ind)
        return population

    def parent_selection(self, population):
        m = int(min(0.5 * len(population), 10))
        fitness_values = [ind['objective'] for ind in population]
        # 计算最小值和最大值
        f_min = min(fitness_values)
        f_max = max(fitness_values)
        # 防止除零：若所有适应度相同，则使用均匀概率
        if f_max == f_min:
            probs = [1.0 / len(fitness_values)] * len(fitness_values)
        else:
            # 最小-最大归一化
            norm_fitness = [(f - f_min) / (f_max - f_min) for f in fitness_values]
            total = sum(norm_fitness)
            probs = [nf / total for nf in norm_fitness]
        parents = random.choices(population, weights=probs, k=m)
        return parents

    def no_cluster_test(self):
        clusters = {}
        for i in range(72):
            clusters[i+1] = [i]
        return clusters



