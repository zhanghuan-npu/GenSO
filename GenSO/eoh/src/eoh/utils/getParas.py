"""
Paras 参数管理类

本模块定义了 `Paras` 类，用于统一管理进化算法或搜索实验的所有参数。
该类提供默认参数、动态更新方法以及自动初始化逻辑，确保算法在不同
问题和方法下可以顺利运行。Paras 类在整个框架中充当“配置中心”的角色，
供 EVOL、AEL、EOH、LS 等算法调用。

功能说明
--------
1. 基本实验设置：
   - `method`：优化算法类型，如 'eoh'、'ael'、'ls' 等。
   - `problem`：问题名称或自定义问题对象。
   - `selection`：个体选择算子，可自动根据方法类型初始化。
   - `management`：种群管理策略，可自动根据方法类型初始化。

2. 进化计算 (EC) 参数：
   - `ec_pop_size`：种群规模。
   - `ec_n_pop`：进化代数。
   - `ec_operators`：进化操作符列表，如 ['e1','e2','m1','m2']。
   - `ec_m`：父代数量。
   - `ec_operator_weights`：操作符使用概率。

3. 大语言模型 (LLM) 设置：
   - 支持本地或远程模型。
   - 可配置 API 地址、密钥和模型类型。

4. 实验控制：
   - 调试模式、输出路径、种子初始化、断点续跑、并行进程数量等。

5. 评估设置：
   - 单个算法评估超时时间。
   - 是否启用 `numba` 加速评估函数。

6. 参数初始化与更新方法：
   - `set_parallel()`：根据 CPU 核心数设置合理的并行数量。
   - `set_ec()`：根据算法类型自动初始化选择算子、管理策略和操作符。
   - `set_evaluation()`：根据问题类型设置默认评估参数。
   - `set_paras(**kwargs)`：统一接口批量更新参数，同时调用上述初始化方法。
"""
class Paras:
    def __init__(self):
        #####################
        ### General settings  ###
        #####################
        self.method = 'eoh'
        self.problem = 'tsp_construct'
        self.selection = None
        self.management = None

        #####################
        ###  EC settings  ###
        #####################
        self.ec_pop_size = 5  # number of algorithms in each population, default = 10
        self.ec_n_pop = 5 # number of populations, default = 10
        self.ec_operators = None # evolution operators: ['e1','e2','m1','m2'], default =  ['e1','e2','m1','m2']
        self.ec_m = 2  # number of parents for 'e1' and 'e2' operators, default = 2
        self.ec_operator_weights = None  # weights for operators, i.e., the probability of use the operator in each iteration, default = [1,1,1,1]
        
        #####################
        ### LLM settings  ###
        #####################
        self.llm_api_endpoint = None # endpoint for remote LLM, e.g., api.deepseek.com
        self.llm_api_key = None  # API key for remote LLM, e.g., sk-xxxx
        self.llm_model = None  # model type for remote LLM, e.g., deepseek-chat

        #####################
        ###  Exp settings  ###
        #####################
        self.exp_debug_mode = False  # if debug
        self.exp_output_path = "./"  # default folder for ael outputs
        self.exp_use_seed = False
        self.exp_seed_path = "./seeds/seeds.json"
        self.exp_use_continue = False
        self.exp_continue_id = 0
        self.exp_continue_path = "./results/pops/population_generation_0.json"
        self.exp_n_proc = 1 # 并行进程数
        
        #####################
        ###  Evaluation settings  ###
        #####################
        self.eva_timeout = 100
        self.eva_numba_decorator = False


    def set_parallel(self):
        import multiprocessing
        num_processes = multiprocessing.cpu_count()
        if self.exp_n_proc == -1 or self.exp_n_proc > num_processes:
            self.exp_n_proc = num_processes
            print(f"Set the number of proc to {num_processes} .")
    
    def set_ec(self):    
        
        if self.management == None:
            if self.method in ['ael','eoh']:
                self.management = 'pop_greedy'
            elif self.method == 'ls':
                self.management = 'ls_greedy'
            elif self.method == 'sa':
                self.management = 'ls_sa'
            else:
                self.management = 'pop_greedy'
        
        if self.selection == None:
            self.selection = 'prob_rank'
            
        
        if self.ec_operators == None:
            if self.method == 'eoh':
                self.ec_operators  = ['e1','e2','m1','m2']
            elif self.method == 'ael':
                self.ec_operators  = ['crossover','mutation']
            elif self.method == 'ls':
                self.ec_operators  = ['m1']
            elif self.method == 'sa':
                self.ec_operators  = ['m1']

        if self.method in ['ls','sa'] and self.ec_pop_size >1:
            self.ec_pop_size = 1
            self.exp_n_proc = 1
            print("> single-point-based, set pop size to 1. ")
            
    def set_evaluation(self):
        # Initialize evaluation settings
        if self.problem == 'bp_online':
            self.eva_timeout = 600
        elif self.problem == 'tsp_construct':
            self.eva_timeout = 600
                
    def set_paras(self, *args, **kwargs):
        
        # Map paras
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
              
        # Identify and set parallel 
        self.set_parallel()
        
        # Initialize method and ec settings
        self.set_ec()
        
        # Initialize evaluation settings
        self.set_evaluation()
