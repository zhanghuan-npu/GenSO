"""
New Method: 基于大语言模型 (LLM) 的进化优化框架

本模块定义了 `LPBS` 类，用于运行基于 LLM 的进化计算流程。该框架
结合 LLM 的生成能力与进化算法的搜索机制，实现了自动化的启发式方法演化。

功能说明
--------
1. 类初始化：
   - 读取实验参数，包括 LLM 接口、种群大小、进化代数、调试模式等
   - 配置是否使用种子个体或继续运行已有种群
   - 初始化 LLM 接口对象 Evolution，用于生成/修复个体

2. 主运行流程 (run)：
   - 初始化种群（新建、加载或使用种子）
   - 评估初始种群适应度
   - 在设定的迭代次数内循环：
       * 计算归一化分数
       * 选择父代
       * 通过交叉和变异生成子代
       * 评估子代并加入种群
       * 种群管理，保留最优个体
       * 保存当前种群和最优个体到文件
   - 输出运行进度与结果

关键特性
--------
- 与 LLM 交互生成个体代码（启发式方法）
- 动态选择机制，避免种群早熟
- Pilot Run 与修复机制，确保个体代码可执行
- 支持从已有种群继续运行或从种子开始
- 自动保存运行过程的中间结果和最优解
"""
import json
import random
import time
from .lpbs_llm import Evolution
from .lpbs_ec import InterfaceEC
from .dynamicSeclection import dynamic_selection

class NewMethod:
    def __init__(self, paras, problem, select, manage, **kwargs):

        self.prob = problem
        self.select = select
        self.manage = manage

        # LLM settings
        self.api_endpoint = paras.llm_api_endpoint
        self.api_key = paras.llm_api_key
        self.llm_model = paras.llm_model

        # Experimental settings
        self.pop_size = paras.ec_pop_size  # population size
        self.n_pop = paras.ec_n_pop  # number of generations

        self.debug_mode = paras.exp_debug_mode  # if debug
        self.ndelay = 1  # default

        self.interface_llm = Evolution(self.api_endpoint, self.api_key, self.llm_model, self.debug_mode, self.prob.prompts)

        self.use_seed = paras.exp_use_seed
        self.seed_path = paras.exp_seed_path
        self.load_pop = paras.exp_use_continue
        self.load_pop_path = paras.exp_continue_path
        self.load_pop_id = paras.exp_continue_id

        self.output_path = paras.exp_output_path

        self.exp_n_proc = paras.exp_n_proc

        self.timeout = paras.eva_timeout

        self.use_numba = paras.eva_numba_decorator

        print("- EoH parameters loaded -")

        # Set a random seed
        random.seed(2024)

    def run(self, job=1):
        """
        主运行函数，用于执行 New Method 进化算法流程

        参数:
            job (int): 当前任务编号，用于文件命名和保存

        返回:
            None
        """
        print("- New Method Evolution Start -")

        time_start = time.time() # 记录起始时间

        # 评估接口（问题相关）
        interface_prob = self.prob

        # 进化算子接口（交叉、变异、修复等）
        interface_ec = InterfaceEC(self.pop_size, self.api_endpoint, self.api_key, self.llm_model,
                                   self.debug_mode, interface_prob, timeout=self.timeout)

        # 初始化
        population = []
        if self.use_seed:  # 读取种子数据
            print("load seeds from " + self.seed_path)
            with open(self.seed_path) as file:
                data = json.load(file)
            population = interface_ec.init_population(self.pop_size)
            for individual in data:
                population.append(individual)
            filename = self.output_path + f"/results_{job}/pops/population_generation_0.json"
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)
        else:
            if self.load_pop:  # 从文件加载种群
                print("load initial population from " + self.load_pop_path)
                with open(self.load_pop_path) as file:
                    data = json.load(file)
                for individual in data:
                    population.append(individual)
                print("initial population has been loaded!")
            else:  # 创建新种群
                print("creating initial population:")
                population = interface_ec.diverse_initialization(self.pop_size)
                print("initial population has been created!")

        # 种子
        # with open(r"C:\Users\张桓\PycharmProjects\EoH-main\eoh\src\eoh\problems\asp\seeds.json", "r", encoding="utf-8") as f:
        #    seeds = json.load(f)
        # population.extend(seeds)
        print(f"初始种群规模为 {len(population)}")

        # 对初始种群进行评估
        population, evals = interface_ec.evaluate_population(population)
        population = interface_ec.population_management(population, [])
        # print("初始种群信息：")
        # for idx, ind in enumerate(population, start=1):
        #     print(f"Ind {idx}: {ind.get('objective', None)}")
        # 保存评估后的初始种群
        filename = self.output_path + f"/results_{job}/pops/population_generation_0.json"
        with open(filename, 'w') as f:
            json.dump(population, f, indent=5)

        # 迭代进化，直到达到最大代数
        for iteration in range(1, self.n_pop + 1):
            # 动态选择
            parents = dynamic_selection(population, iteration, self.n_pop)
            parents = [population[idx] for idx in parents]
            # 父母排序
            parents.sort(key=lambda ind: ind['objective'])
            # 案例聚类
            clusters = interface_ec.cluster_cases(parents)
            # 提取模式
            modes = interface_ec.extract_modes(parents, clusters)
            # 产生后代
            offsprings = interface_ec.crossover_population(parents, modes, self.pop_size)
            # 种群管理
            population = interface_ec.population_management(population, offsprings)
            # 产生后代
            offsprings = interface_ec.mutate_population(population, clusters, modes, self.pop_size)
            # 种群管理
            population = interface_ec.population_management(population, offsprings)
            # 找到精英
            best = interface_ec.get_best_individual(population)

            # 保存当前代种群
            filename = self.output_path + f"/results_{job}/pops/population_generation_" + str(iteration) + ".json"
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)

            # 保存当前代的最优个体
            filename = self.output_path + f"/results_{job}/pops_best/population_generation_" + str(iteration) + ".json"
            with open(filename, 'w') as f:
                json.dump(best, f, indent=5)

            # 将长期反思保存到文件
            filename_lt = self.output_path + f"/results_{job}/mode_descriptions_{iteration}.json"
            with open(filename_lt, 'w') as f:
                json.dump(modes, f, indent=5)

            # 输出进度与结果
            print(f"--- {iteration} of {self.n_pop} populations finished. Time Cost:  {((time.time()-time_start)/60):.1f} m")
            print("Pop Objs: ", end=" ")
            # for i in range(len(population)):
            #     print(str(population[i]['objective']) + " ", end="")
            print()