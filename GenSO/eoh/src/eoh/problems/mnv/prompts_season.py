from .training_case import TrainingCases


class GetPrompts:
    """
    针对多期报童问题的 Prompt 类
    设计思路：
        1. 提供详细问题描述，帮助模型理解多期报童问题背景、目标及约束条件。
        2. 指定自动生成函数的名称、输入、输出格式。
        3. 提供额外信息模块，包含输入数据结构、示例及说明。
    """
    def __init__(self):
        # --- 1. 任务说明 ---
        # 让模型理解问题本身，而不涉及函数生成细节
        self.prompt_task = (
            "I need help designing the mapping of basis functions for a generalized decision rule in a multi-period "
            "newsvendor problem. The model utilizes Sample Average Approximation based on historical demand scenarios, "
            "where the wait-and-see replenishment order is approximated by a decision rule parameterized by period-"
            "specific basis functions. The objective is to determine the optimal here-and-now warehouse capacity and "
            "the coefficients of the replenishment policy to minimize the total expected cost."
        )
        # --- 2. 额外问题描述模块 ---
        # 描述预约调度问题的背景、目标、性能指标和关键参数
        self.problem_name = "multi-period newsvendor problem"
        self.problem_description = (
            "The multi-period newsvendor problem involves a single product managed over $T$ periods. You must determine "
            "the optimal initial capacity $q$ and a replenishment policy $y_t(D_t)$ under uncertain demand $d_t$, where "
            "$D_t$ represents the demand history $(d_1, ..., d_{t-1})$. The replenishment quantity $y_t$ is decided "
            "at the start of period $t$, while the demand $d_t$ is realized at the end of period $t$."
            "We build a stochastic programming model using Sample Average Approximation with $S$ training scenarios. "
            "The model is implemented using Gurobi. The parameters are:\n"
            "T = 24\n"
            "Unit capacity cost: c_q = 5\n"
            "Unit shortage cost: c_s = 10"
            "Unit holding cost: c_h = 1"
            "Initial inventory: v_0 = 0"
            "Demand $d_{st}$ is sampled from a stochastic process."
            "nominal demand: d_t^0 = 1000 * (1 + 12 * sin( π(t − 1)/12 ) )\n"
            "The objective is to minimize:\n"
            "$c_q * q + frac{1}{S} \sum_{s=1}^{S} \sum_{t=1}^{T} (c_s * xi_{st}^+ + c_h * xi_{st}^-)$\n"
            "subject to:\n"
            "Inventory Balance: $v_{st} = v_{s,t-1} + y_{st} - d_{st}, forall s, t$"
            "Shortage Linearization: $xi_{st}^+ \geq d_{st} - v_{s,t-1} - y_{st}, forall s, t$"
            "Holding Linearization: $xi_{st}^- \geq v_{s,t-1} + y_{st} - d_{st}, forall s, t$"
            "Capacity Constraints: $0 \leq v_{st} \leq q$ and $0 \leq y_{st} \leq q, forall s, t$"
            "Decision Rule Policy: $y_{st} = y_{t,0} + \sum_{h \in [H_t]} y_{t,h} * phi_{t,h}(\hat{D}_{st}), forall s, t$"
            "The decision variable $y_{st}$ is not a free variable; it is constrained by a generalized decision rule "
            "where $y_{t,0}$ is the intercept and $y_{t,h}$ are the coefficients applied to the basis functions "
            "$\phi_{t,h}$. These features $\phi_{t,h}$ must be derived solely from the demand history $D_t$. Your task "
            "is to design a complex, high-dimensional composite basis function structure $\Phi_t$ for the features.\n"
            "Be Imaginative: Use any combination of statistical indicators (rolling windows, exponential smoothing, "
            "volatility), mathematical kernels (Fourier series for periodicity, Legendre polynomials, Sigmoid or "
            "ReLU-like non-linear mappings), and problem-specific indicators (remaining time to horizon, cost ratios "
            "$c_s/c_h$).\n"
            "Adaptive Logic: Design distinct feature sets for different stages of the horizon (e.g., initial ramp-up vs. "
            "steady-state vs. end-of-horizon liquidation) and ensure the features capture the non-linear dependencies "
            "between past demand and optimal future replenishment. "
        )

        # --- 3. 自动生成函数格式 ---
        # 指定函数名称和输入输出要求，方便模型生成符合接口的函数
        self.prompt_func_name = "get_features"
        self.prompt_func_inputs = ['t', 'd']
        self.prompt_func_outputs = ['features']
        self.prompt_func_signature = "def get_features(t: int, d: np.ndarray) -> list:"
        self.prompt_func_example = (
            "import numpy as np\n"
            "import pandas as pd\n"
            "def get_features(t: int, d: np.ndarray):\n"
            "    # 示例逻辑：组合原始需求、移动平均与非线性项\n"
            "    features = []\n"
            "    if t > 0:\n"
            "        # 基础项：前一期需求\n"
            "        features.append(d[t-1])\n"
            "        # 统计项：过去3期的平均需求\n"
            "        lookback = 3\n"
            "        avg_d = np.mean(d[max(0, t-lookback):t])\n"
            "        features.append(avg_d)\n"
            "        # 非线性项：需求的平方项，用于捕捉波动的非线性影响\n"
            "        features.append(d[t-1]**2)\n"
            "        # 周期项：利用正弦函数模拟可能的季节性反馈\n"
            "        features.append(np.sin(2 * np.pi * t / 12) * d[t-1])\n"
            "    return features\n"
        )

        # --- 4. 输入输出说明 ---
        # 明确输入参数的类型和意义，输出值的含义
        self.prompt_inout_inf = (
            "Inputs:\n"
            "t: An integer (0 ~ T-1) representing the current time period index.\n"
            "d: A 1D NumPy ndarray of length T, representing a single realized scenario of demand.\n"
            "Output:\n"
            "features: A Python list of numerical values. Each value represents a basis function or a transformed "
            "feature derived from historical demand d[:t]."
        )
        # --- 5. 额外要求模块 ---
        # 提供明确要求
        self.prompt_other_inf = (
            "The following constraints and guidelines must be strictly followed to ensure the generated feature "
            "extraction function is mathematically sound and compatible with the SAA-LDR framework:"
            "Self-Contained Imports: If your implementation relies on any external libraries (such as numpy, scipy, or "
            "pandas), you must explicitly include the necessary import statements within the generated code block to "
            "ensure the function is fully self-contained and executable."
            "Strict Non-Anticipativity: The production decision at time t must only depend on historical realizations. "
            "Therefore, get_features(t, d) must only access elements of d from index 0 to t-1. Any access to d[t] "
            "or beyond is a violation of causality and is strictly prohibited."
            "Handling Initial States: At t=0, since no prior demand information is available, the function should "
            "typically return an empty list []. In this case, the decision y_{0} will automatically be treated as "
            "a non-adaptive constant."
            "Numerical Stability and Scaling: When using high-order polynomial terms (e.g., $d^2, d^3$) or exponential "
            "functions, apply appropriate scaling factors (e.g., dividing by a constant or the mean of past data) to "
            "prevent large numerical ranges that could destabilize the optimization solver."
            "Feature Diversity: You are encouraged to combine multiple types of information, such as moving averages for "
            "smoothing, rolling variance for volatility, or trigonometric functions for seasonality, to capture the "
            "complex dynamics of the inventory problem."
            "Dimensionality Constraint: To maintain computational efficiency and prevent overfitting, the feature list "
            "returned by get_features(t, d) must be concise; its total length must never exceed 10 elements for any "
            "given period $t$."
            "Do not use any random component. Do not provide any explanations or additional text except the Python code. "
        )
        # --- 6. 外部知识 ---
        # 作为Reevo方法的初始长期反思消息
        self.prompt_external_knowledge = ""

        # --- 7. 知识提取任务 ---
        # LPSE方法的知识提取提示词
        self.prompt_knowledge_extraction_task = (
            "The following dataset consists of multiple demand realizations (scenarios) over 24 periods. "
            "Your first task is to perform a deep statistical audit of these scenarios to identify 'feature-rich' "
            "indicators. Beyond calculating mean and variance for each period, you must detect non-linear "
            "temporal dependencies, such as heteroskedasticity (volatility clusters) and potential seasonal "
            "oscillations. Identify which lagged periods (from t-1 to t-4) or rolling windows show the "
            "strongest predictive power for future inventory strain."
        )
        self.case_characteristics = TrainingCases()
        self.prompt_expert_analyze_task = (
            "Next, analyze the logic of an expert feature extraction algorithm that achieves superior performance "
            "by directly aligning its structure with the identified demand profiles. You must summarize how the "
            "algorithm translates the observed demand realizations, scenario-based variability, and time-dependent cost "
            "structure into the algebraic form of the linear decision rules p."
            "Provide precise, actionable guidance on how these statistical insights and the non-anticipativity "
            "requirements dictate the adaptation scope of p to optimize the trade-off between inventory holding costs "
            "and production capacity utilization."
        )

        # --- 8. 多样化初始任务 ---
        self.diverse_initialization_prompt = [
            (
                "The multi-period newsvendor problem involves a single product managed over $T$ periods. You must determine "
                "the optimal initial capacity $q$ and a replenishment policy $y_t(D_t)$ under uncertain demand $d_t$, where "
                "$D_t$ represents the demand history $(d_1, ..., d_{t-1})$. The replenishment quantity $y_t$ is decided "
                "at the start of period $t$, while the demand $d_t$ is realized at the end of period $t$."
                "We build a stochastic programming model using Sample Average Approximation with $S$ training scenarios. "
                "The model is implemented using Gurobi. The parameters are:\n"
                "T = 24\n"
                "Unit capacity cost: c_q = 5\n"
                "Unit shortage cost: c_s = 10"
                "Unit holding cost: c_h = 1"
                "Initial inventory: v_0 = 0"
                "Demand $d_{st}$ is sampled from a stochastic process."
                "nominal demand: d_t^0 = 1000 * (1 + 12 * sin( π(t − 1)/12 ) )\n"
                "The objective is to minimize:\n"
                "$c_q * q + frac{1}{S} \sum_{s=1}^{S} \sum_{t=1}^{T} (c_s * xi_{st}^+ + c_h * xi_{st}^-)$\n"
                "subject to:\n"
                "Inventory Balance: $v_{st} = v_{s,t-1} + y_{st} - d_{st}, forall s, t$"
                "Shortage Linearization: $xi_{st}^+ \geq d_{st} - v_{s,t-1} - y_{st}, forall s, t$"
                "Holding Linearization: $xi_{st}^- \geq v_{s,t-1} + y_{st} - d_{st}, forall s, t$"
                "Capacity Constraints: $0 \leq v_{st} \leq q$ and $0 \leq y_{st} \leq q, forall s, t$"
                "Decision Rule Policy: $y_{st} = y_{t,0} + \sum_{h \in [H_t]} y_{t,h} * phi_{t,h}(\hat{D}_{st}), forall s, t$"
                "The decision variable $y_{st}$ is not a free variable; it is constrained by a generalized decision rule "
                "where $y_{t,0}$ is the intercept and $y_{t,h}$ are the coefficients applied to the basis functions "
                "$\phi_{t,h}$. These features $\phi_{t,h}$ must be derived solely from the demand history $D_t$. Your task "
                "is to design a complex, high-dimensional composite basis function structure $\Phi_t$ for the features.\n"
                "Be Imaginative: Use any combination of statistical indicators (rolling windows, exponential smoothing, "
                "volatility), mathematical kernels (Fourier series for periodicity, Legendre polynomials, Sigmoid or "
                "ReLU-like non-linear mappings), and problem-specific indicators (remaining time to horizon, cost ratios "
                "$c_s/c_h$).\n"
                "Adaptive Logic: Design distinct feature sets for different stages of the horizon (e.g., initial ramp-up vs. "
                "steady-state vs. end-of-horizon liquidation) and ensure the features capture the non-linear dependencies "
                "between past demand and optimal future replenishment. "
            )
        ]

    # --- Getter 方法 ---
    def get_init_system(self):
        return self.init_system

    def get_task(self):
        return self.prompt_task

    def get_problem_name(self):
        return self.problem_name

    def get_problem_description(self):
        return self.problem_description

    def get_func_name(self):
        return self.prompt_func_name

    def get_func_inputs(self):
        return self.prompt_func_inputs

    def get_func_outputs(self):
        return self.prompt_func_outputs

    def get_func_signature(self):
        return self.prompt_func_signature

    def get_func_example(self):
        return self.prompt_func_example

    def get_inout_inf(self):
        return self.prompt_inout_inf

    def get_other_inf(self):
        return self.prompt_other_inf

    def get_external_knowledge(self):
        # return self.prompt_external_knowledge
        return ""

    def get_prompt_knowledge_extraction_task(self):
        return self.prompt_knowledge_extraction_task

    def get_case_characteristics(self):
        return self.case_characteristics

    def get_prompt_expert_analyze_task(self):
        return self.prompt_expert_analyze_task

    def get_diverse_initialization_prompt(self):
        return self.diverse_initialization_prompt