from .training_case import TrainingCases


class GetPrompts:
    def __init__(self):
        # --- 1. 任务说明 ---
        # 让模型理解问题本身，而不涉及函数生成细节
        self.prompt_task = (
            "I need help designing the mapping of basis functions for a generalized decision rule in a data center "
            "network planning problem. The model utilizes Sample Average Approximation based on historical demand scenarios, "
            "where the wait-and-see replenishment order is approximated by a decision rule parameterized by period-"
            "specific basis functions. The objective is to determine the optimal here-and-now data center locations, "
            "installed capacities, and demand assignments, as well as the coefficients of the allocation policy to "
            "minimize the total expected investment and operational costs. "
        )
        # --- 2. 额外问题描述模块 ---
        # 描述预约调度问题的背景、目标、性能指标和关键参数
        self.problem_name = "Data Center Facility Location and Adaptive Job Scheduling"
        self.problem_description = (
            "The data center network planning problem involves coordinating facility location, capacity installation, "
            "and demand assignment over $T$ periods. There are $I$ candidate data center locations and $J$ demand "
            "locations, each generating stochastic computation jobs. You must determine the optimal binary location "
            "variables $x_i$, capacity $u_i$, and assignment $a_{ij}$ under uncertain computation demand $d_{jt}$, "
            "alongside a dynamic allocation policy $y_{ijt}(D_{[t-1]})$ that maps demand history to processing power. "
            "The allocation $y_{ijt}$ is decided at the beginning of period $t$ based on observed history, while "
            "$d_{jt}$ is realized throughout the period. We build a stochastic programming model using Sample Average "
            "Approximation with $S$ training scenarios, implemented in Gurobi. The parameters are:\n"
            "T = 2\n"
            "Fixed setup cost $f_i$\n"
            "Unit capacity cost: cu_i\n"
            "Dispatch cost: ca_ij\n"
            "Unit delay penalty: cl = 1\n"
            "Unit outsourcing cost: co = 5\n"
            "The objective is to minimize:\n"
            "$sum (f_i x_i + cu_i u_i) + sum ca_{ij} a_{ij} + frac{1}{S} sum_{s=1}^{S} (sum_{t=1}^{T} sum_{j=1}^{J} cl \ell_{sjt} + sum_{j=1}^{J} co o_{sjT})$\n"
            "subject to:\n"
            "Backlog Evolution: $l_{sjt} = l_{s,j,t-1} + d_{sjt} - \sum_i a_{ij} y_{sijt}, forall s, j, t$\n"
            "Final Period Outsourcing: $l_{sj,T-1} + d_{sjT} - \sum_i a_{ij} y_{sijT} - o_{sjT} = 0, forall s, j$\n"
            "Capacity Constraints: $\sum_j a_{ij} y_{sijt} \leq u_i, forall s, i, t$\n"
            "Assignment Constraints: $\sum_i a_{ij} = 1$ and $a_{ij} \leq x_i, forall i, j$\n"
            "Decision Rule Policy: $y_{sijt} = y^0_{ijt} + \sum_{h \in [H_t]} y^h_{ijt} \phi_{t,h}(\hat{D}_{s,[t-1]}), \forall s, i, j, t$\n"
            "The decision variable $y_{sijt}$ is not a free variable; it is constrained by a generalized decision rule "
            "where $y_{ijt,0}$ is the intercept and $y_{ijt,h}$ are the coefficients applied to the basis functions "
            "$\phi_{ijt,h}$. These features $\phi_{ijt,h}$ must be derived solely from the demand history $D_t-1$. "
            "Your task is to design a complex, high-dimensional composite basis function structure $\Phi_ijt$ for the features.\n"
            "Be Imaginative: Use combinations of statistical indicators like rolling average workloads and demand "
            "volatility, mathematical kernels including Fourier series for diurnal cycles or radial basis functions, "
            "and problem-specific indicators such as the critical ratio $cl/co$ or proximity-weighted aggregate demand."
            "Adaptive Logic: Design distinct feature sets for different horizon stages, such as initial cold-start "
            "periods versus end-of-horizon backlog clearing, ensuring the features capture non-linear dependencies "
            "between cumulative historical load and optimal future capacity allocation."
        )

        # --- 3. 自动生成函数格式 ---
        # 指定函数名称和输入输出要求，方便模型生成符合接口的函数
        self.prompt_func_name = "get_features"
        self.prompt_func_inputs = ['i', 'j', 't', 'ca', 'T', 'D_s']
        self.prompt_func_outputs = ['features']
        self.prompt_func_signature = "def get_features(i: int, j: int, t: int, ca: np.ndarray, T: int, D_s: np.ndarray) -> list:"
        self.prompt_func_example = (
            "def get_features(i, j, t, ca, T, D_s):\n"
            "    # Example logic: Capturing historical load, network proximity, and temporal urgency\n"
            "    if t == 0:\n"
            "        return []\n"
            "\n"
            "    features = []\n"
            "    # 1. Local Accumulation: Total workload backlog potential at demand location j\n"
            "    phi_cum = np.sum(D_s[j, :t])\n"
            "    features.append(phi_cum)\n"
            "\n"
            "    # 2. Network Load: Proximity-weighted demand from other locations assigned to center i\n"
            "    # Represents the potential resource competition within the same data center\n"
            "    phi_network = np.sum(ca[i, :] * D_s[:, t-1])\n"
            "    features.append(phi_network)\n"
            "\n"
            "    # 3. Workload Volatility: Change in demand at location j to capture bursty traffic\n"
            "    phi_momentum = D_s[j, t-1] - D_s[j, t-2] if t >= 2 else 0.0\n"
            "    features.append(phi_momentum)\n"
            "\n"
            "    # 4. Service Urgency: Linear decay representing the remaining horizon to prioritize backlog clearing\n"
            "    phi_urgency = (T - t + 1) / T\n"
            "    features.append(phi_urgency)\n"
            "\n"
            "    return features\n"
        )

        # --- 4. 输入输出说明 ---
        # 明确输入参数的类型和意义，输出值的含义
        self.prompt_inout_inf = (
            "Inputs:\n"
            "i: An integer representing the data center index.\n"
            "j: An integer representing the demand location index.\n"
            "t: An integer (0 ~ T-1) representing the current time period index.\n"
            "ca: A 2D NumPy ndarray (I x J) representing the current assignment matrix.\n"
            "T: An integer representing the total scheduling horizon.\n"
            "D_s: A 2D NumPy ndarray (J x T) representing the full realized demand scenario across all locations.\n"
            "Output:\n"
            "features: A Python list of numerical values. Each value represents a basis function or a transformed "
            "feature derived from the demand history D_s[:, :t] and network parameters to parameterize the adaptive "
            "allocation decision rule."
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
            "Therefore, get_features(i, j, t, ca, T, D_s) must only access elements of d from index 0 to t-1. Any access to D_s[:,t] "
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
            "returned by get_features(i, j, t, ca, T, D_s) must be concise; its total length must never exceed 5 elements for any "
            "given period $t$."
            "Do not use any random component. Do not provide any explanations or additional text except the Python code. "
        )
        # --- 6. 外部知识 ---
        # 作为Reevo方法的初始长期反思消息
        self.prompt_external_knowledge = ""

        # --- 7. 知识提取任务 ---
        # LPSE方法的知识提取提示词
        self.prompt_knowledge_extraction_task = (
            "The following dataset consists of multiple demand realizations (scenarios) over 16 periods. "
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
                "The data center network planning problem involves coordinating facility location, capacity installation, "
                "and demand assignment over $T$ periods. There are $I$ candidate data center locations and $J$ demand "
                "locations, each generating stochastic computation jobs. You must determine the optimal binary location "
                "variables $x_i$, capacity $u_i$, and assignment $a_{ij}$ under uncertain computation demand $d_{jt}$, "
                "alongside a dynamic allocation policy $y_{ijt}(D_{[t-1]})$ that maps demand history to processing power. "
                "The allocation $y_{ijt}$ is decided at the beginning of period $t$ based on observed history, while "
                "$d_{jt}$ is realized throughout the period. We build a stochastic programming model using Sample Average "
                "Approximation with $S$ training scenarios, implemented in Gurobi. The parameters are:\n"
                "T = 2\n"
                "Fixed setup cost $f_i$\n"
                "Unit capacity cost: cu_i\n"
                "Dispatch cost: ca_ij\n"
                "Unit delay penalty: cl = 1\n"
                "Unit outsourcing cost: co = 5\n"
                "The objective is to minimize:\n"
                "$sum (f_i x_i + cu_i u_i) + sum ca_{ij} a_{ij} + frac{1}{S} sum_{s=1}^{S} (sum_{t=1}^{T} sum_{j=1}^{J} cl \ell_{sjt} + sum_{j=1}^{J} co o_{sjT})$\n"
                "subject to:\n"
                "Backlog Evolution: $l_{sjt} = l_{s,j,t-1} + d_{sjt} - \sum_i a_{ij} y_{sijt}, forall s, j, t$\n"
                "Final Period Outsourcing: $l_{sj,T-1} + d_{sjT} - \sum_i a_{ij} y_{sijT} - o_{sjT} = 0, forall s, j$\n"
                "Capacity Constraints: $\sum_j a_{ij} y_{sijt} \leq u_i, forall s, i, t$\n"
                "Assignment Constraints: $\sum_i a_{ij} = 1$ and $a_{ij} \leq x_i, forall i, j$\n"
                "Decision Rule Policy: $y_{sijt} = y^0_{ijt} + \sum_{h \in [H_t]} y^h_{ijt} \phi_{t,h}(\hat{D}_{s,[t-1]}), \forall s, i, j, t$\n"
                "The decision variable $y_{sijt}$ is not a free variable; it is constrained by a generalized decision rule "
                "where $y_{ijt,0}$ is the intercept and $y_{ijt,h}$ are the coefficients applied to the basis functions "
                "$\phi_{ijt,h}$. These features $\phi_{ijt,h}$ must be derived solely from the demand history $D_t-1$. "
                "Your task is to design a complex, high-dimensional composite basis function structure $\Phi_ijt$ for the features.\n"
                "Be Imaginative: Use combinations of statistical indicators like rolling average workloads and demand "
                "volatility, mathematical kernels including Fourier series for diurnal cycles or radial basis functions, "
                "and problem-specific indicators such as the critical ratio $cl/co$ or proximity-weighted aggregate demand."
                "Adaptive Logic: Design distinct feature sets for different horizon stages, such as initial cold-start "
                "periods versus end-of-horizon backlog clearing, ensuring the features capture non-linear dependencies "
                "between cumulative historical load and optimal future capacity allocation."
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