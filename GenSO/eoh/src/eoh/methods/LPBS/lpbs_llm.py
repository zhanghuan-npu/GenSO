import ast
import re
from ...llm.interface_api import InterfaceAPI


class Evolution:

    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode, prompts):
        """
        初始化 Evolution 类。

        参数：
            api_endpoint (str): LLM API 地址
            api_key (str): API 密钥
            model_LLM (str): 使用的模型名称
            debug_mode (bool): 调试模式开关
            prompts (object): 提示词对象，提供问题相关描述与函数格式
        """
        # 设置提示接口
        self.prompt_task = prompts.get_task()
        self.prompt_problem_name = prompts.get_problem_name()
        self.prompt_problem_description = prompts.get_problem_description()
        self.prompt_func_name = prompts.get_func_name()
        self.prompt_func_inputs = prompts.get_func_inputs()
        self.prompt_func_outputs = prompts.get_func_outputs()
        self.prompt_func_signature = prompts.get_func_signature()
        self.prompt_inout_inf = prompts.get_inout_inf()
        self.prompt_func_examples = prompts.get_func_example()
        self.prompt_other_inf = prompts.get_other_inf()
        if len(self.prompt_func_inputs) > 1:
            self.joined_inputs = ", ".join("'" + s + "'" for s in self.prompt_func_inputs)
        else:
            self.joined_inputs = "'" + self.prompt_func_inputs[0] + "'"

        if len(self.prompt_func_outputs) > 1:
            self.joined_outputs = ", ".join("'" + s + "'" for s in self.prompt_func_outputs)
        else:
            self.joined_outputs = "'" + self.prompt_func_outputs[0] + "'"

        self.prompt_knowledge_extraction_task = prompts.get_prompt_knowledge_extraction_task()
        self.case_characteristics = prompts.get_case_characteristics()
        self.prompt_expert_analyze_task = prompts.get_prompt_expert_analyze_task()
        self.diverse_initialization_prompt = prompts.get_diverse_initialization_prompt()

        # 设置LLM
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode  # close prompt checking

        self.interface_api = InterfaceAPI(self.api_endpoint, self.api_key, self.model_LLM, self.debug_mode)

    def get_init_prompt(self):
        """
        构造生成初始个体的 Prompt。

        返回：
            tuple: (system, user)，分别为系统提示和用户提示
        """
        # 系统角色提示
        system = (
            "[System]: You are an esteemed expert in operations research and heuristic algorithm design. "
            "Your mission is to invent a novel heuristic algorithm tailored to the given optimization problem."
        )

        # 任务描述
        task = f"[Task]: {self.prompt_task} {self.prompt_problem_description} \n"


        # 函数描述
        func = (
            f"[Function I/O]: You should implement a Python function in the format of: \n"
            f"{self.prompt_func_signature} \n"
            f"The function should accept {len(self.prompt_func_inputs)} inputs: {self.joined_inputs}, "
            f"and should return {len(self.prompt_func_outputs)} output: {self.joined_outputs}.\n"
            f"{self.prompt_inout_inf}"
            f"Example Code:\n {self.prompt_func_examples}"
        )

        # 补充说明
        other = f"[Requirements]: {self.prompt_other_inf}"


        return system, task + func + other

    def get_mode_prompt(self, case_list, specialist):
        # 系统角色提示
        system = (
            "[System]: You are an esteemed expert in operations research and heuristic algorithm design. "
            "Your mission is to analyze problem instances and their top-performing algorithm to learn their "
            "characteristics and extract key patterns."
        )

        # 任务描述
        task = f"[Task]: {self.prompt_knowledge_extraction_task}\n"

        case_lines = []
        for idx, case_idx in enumerate(case_list, start=1):
            case = f"Scenario {idx}: {self.case_characteristics.get_case_characteristic(case_idx)}"
            case_lines.append(case)
        case_lines = "\n".join(case_lines)

        # 专家描述
        expert = f"\n{self.prompt_expert_analyze_task}\n" + specialist['code'] +"\n"

        requirement = (
            "[Requirements]: Summarize the context and policy in the following format:\n"
            "   [Context]: #Fill in the common characteristics of these scenarios (around 30 words)#\n"
            "   [Policy]: #Fill in the key pattern and practical guidance (around 30 words)#\n"
        )

        return system, task + case_lines + expert + requirement

    def get_crossover_prompt(self, parents, modes):
        """
        生成用于指导大语言模型（LLM）进行预约调度规则交叉（crossover）的结构化提示词。

        参数：
            parents (list[dict]): 父母算法列表，每个元素包含 {'code': str}
            modes (list[str]): 对应每个父母的模式列表，每个元素是字符串列表

        返回：
            str: 返回包含以下部分的完整提示词字符串：
        """
        # 系统角色提示
        system = (
            "[System]: You are an esteemed expert in operations research and heuristic algorithm design. "
            "Your mission is to invent a novel heuristic algorithm tailored to the given optimization problem."
        )

        # 任务描述
        task = (
            f"[Task]: {self.prompt_task}"
            f"You are required to reference a series of existing {self.prompt_func_name} implementations and construct a new "
            f"{self.prompt_func_name}. You should analyze each parent code together with its associated knowledge to understand "
            "its decision-making patterns, and then synthesize a new function that effectively combines insights from these examples.\n"
        )

        # 父母描述（算法与对应模式一一对应展示）
        algorithms_lines = []
        for idx, (parent, mode) in enumerate(zip(parents, modes), start=1):
            algorithms_lines.append(
                f"[Algorithm {idx}]:\n{parent['code']}\n"
                f"The knowledge associated with Algorithm {idx} is as follows:\n"
                f"[knowledge {idx}]:\n{mode}\n"
            )
        algorithms_lines = "\n".join(algorithms_lines)

        # 函数描述
        func = (
            f"[Function I/O]: You should implement a Python function in the format of: \n"
            f"{self.prompt_func_signature} \n"
            f"The function should accept {len(self.prompt_func_inputs)} inputs: {self.joined_inputs}, "
            f"and should return {len(self.prompt_func_outputs)} output: {self.joined_outputs}.\n"
            f"{self.prompt_inout_inf}"
            f"Example Code:\n {self.prompt_func_examples}"
        )

        # 补充说明
        other = f"[Requirements]: {self.prompt_other_inf}"

        return system, task + algorithms_lines + func + other

    def get_repair_prompt(self, individual):
        """
        构造修复操作的 Prompt。

        参数：
            individual (dict): 待修复的个体，包含 'code' 与 'traceback_msg'

        返回：
            str: 拼接好的 Prompt 字符串
        """
        # 系统角色提示
        system = (
            "[System]: You are an esteemed expert in operations research and software engineer. "
            "Your mission is to carefully debug and repair a given Python function."
        )

        task = (
            f"[Task]: The provided function for solving {self.prompt_problem_name} failed to pass the tests. "
            f"The following error was encountered:\n{individual['traceback_msg']}\n"
            f"You must analyze the error message and repair the function so that it works correctly."
        )
        code = "[Code]\n" + individual['code'] + "\n"

        requirement = (
            f"[Requirements]: Return a corrected version of the {self.prompt_func_name} function "
            "that preserves the same input/output format and logic structure. "
        ) + self.prompt_other_inf
        return system, task + code + requirement

    def get_mutation_prompt(self, individual, mode):
        """
        构造用于指导 LLM 对给定预约调度规则进行变异（改进）的提示词。

        参数：
            individual (dict): 待改进的个体，包含：
                - 'code': 个体的 Python 函数代码，即现有的预约调度规则
                - 'objective': 该个体整体表现的适应度指标（可选）
            mode (str): 自然语言描述的调度模式，
                        表示该个体在此模式下的表现较差，需要基于此进行改进

        返回：
            str: 拼接好的提示词字符串，包含以下部分：
        """
        # 系统角色提示
        system = (
            "[System]: You are an esteemed expert in operations research and heuristic algorithm design. "
            "Your mission is to invent a novel heuristic algorithm tailored to the given optimization problem."
        )

        # 任务描述
        task = (
            f"[Task]: The following {self.prompt_func_name} has shown relatively poor performance on to the given scenario."
            "Your must analyze why it fail to meet expectations in that scenario and improve the rule accordingly, "
            "while keeping its input and output formats unchanged.\n"
        )

        code = f"[Code]:\n {individual['code']} \n"

        mode = f"The context and policy in which this rule performs poorly are described as follows:\n {mode} \n"

        # 补充说明
        other = f"[Requirements]: {self.prompt_other_inf}"

        return system, task + code + mode + other

    def _get_alg(self, system, user, temp=1.0):
        """
        调用 LLM 获取响应，并从中提取 Python 函数代码。

        参数：
            system (str): 系统提示
            user (str): 用户提示
            temp (float): LLM 采样温度

        返回：
            str: 提取到的 Python 代码
        """
        response = self.interface_api.get_response(system_prompt=system, user_prompt=user, temperature=temp)
        # 去掉开头和结尾的 ```python 或 ``` 标记
        code = response.replace("```python", "").replace("```", "").strip()

        try:
            tree = ast.parse(code)
            # 优先提取第一个函数
            func_code = None
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    func_code = ast.get_source_segment(code, node)
                    break
            if func_code:
                code = func_code
        except SyntaxError:
            # AST 解析失败，使用正则退路
            match = re.search(r"(import\s.*|def\s.*:)[\s\S]*", code, re.MULTILINE)
            if match:
                code = match.group(0)
            else:
                # 都没有匹配到，返回原始 code
                code = code

        return code

    def new_individual(self):
        """
        individual (dict): 包含至少以下字段的个体：
            - 'code'         : LLM 生成的 Python 代码字符串
        """
        individual = {'code': None}

        system, user = self.get_init_prompt()

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm: \n", system, user)
            print(">>> Press 'Enter' to continue")
            input()

        code = self._get_alg(system, user, temp=1.3)

        if self.debug_mode:
            print("\n >>> check designed code: \n", code)
            print(">>> Press 'Enter' to continue")
            input()

        individual['code'] = code

        return individual

    def get_mode(self, case_list, specialist):
        """
            基于给定的场景列表（case_list）和表现最佳的专家（specialist），构建提示词（Prompt）并调用 LLM 来生成场景模式（mode）。

            参数：
                case_list (list): 包含多个预约调度场景的列表，每个元素是一个元组 (m, cv, pn, pw, cr)，
                分别表示平均服务时间、变异系数、爽约概率、插队患者概率以及成本比。
                specialist (dict): 在这些场景中表现最优的专家个体，包含其算法代码（code）及适应度等信息。

            返回：
                mode (str): LLM 基于输入场景与专家总结提炼出的预约调度模式（mode），描述这些场景的共同特征及所需调度规则的能力。
        """
        # 构建 LLM Prompt
        system, user = self.get_mode_prompt(case_list, specialist)

        # 调试模式下打印 Prompt
        if self.debug_mode:
            print("\n >>> check prompt for knowledge extraction: \n", system, user)
            print(">>> Press 'Enter' to continue")
            input()

        # 调用 LLM 生成代码
        mode = self.interface_api.get_response(system_prompt=system, user_prompt=user)

        # 调试模式下打印生成代码
        if self.debug_mode:
            print("\n >>> check extracted knowledge: \n", mode)
            print(">>> Press 'Enter' to continue")
            input()

        return mode

    def crossover(self, parents, modes):
        """
        基于父代个体及其对应的调度模式（parents_modes）进行交叉操作，生成一个新的预约调度规则个体。

        参数：
            parents_modes (dict): 字典格式，其中键为父代算法代码（str），值为该算法对应的模式列表（list of str）。

        返回：
            individual (dict): 新生成的个体，包含：
            - 'code': LLM 生成的预约调度函数代码（str）。
        """
        # 初始化个体
        individual = {'code': None}

        # 构建 LLM Prompt
        system, user= self.get_crossover_prompt(parents, modes)

        # 调试模式下打印 Prompt
        if self.debug_mode:
            print("\n >>> check prompt for crossover: \n", system, user)
            print(">>> Press 'Enter' to continue")
            input()

        # 调用 LLM 生成代码
        code = self._get_alg(system, user)

        # 调试模式下打印生成代码
        if self.debug_mode:
            print("\n >>> check crossover code: \n", code)
            print(">>> Press 'Enter' to continue")
            input()

        individual['code'] = code

        return individual

    def repair(self, individual):
        """
        修复无法通过测试的个体。

        参数：
            individual (dict): 待修复的个体，包含 'code' 和 'traceback_msg'

        返回：
            dict: 新修复个体，包含 'code'
        """
        # 初始化个体
        new_individual = {'code': None}

        # 构建 LLM Prompt
        system, user = self.get_repair_prompt(individual)

        # 调试模式下打印 Prompt
        if self.debug_mode:
            print("\n >>> check prompt for repair: \n", system, user)
            print(">>> Press 'Enter' to continue")
            input()

        # 调用 LLM 生成变异代码
        code = self._get_alg(system, user)

        # 调试模式下打印生成代码
        if self.debug_mode:
            print("\n >>> check repaired code: \n", code)
            print(">>> Press 'Enter' to continue")
            input()

        # 保存代码到新个体
        new_individual['code'] = code

        return new_individual

    def mutate(self, individual: dict, mode: str):
        """
        基于一个个体生成一个变异个体。

        参数：
            individual (dict): 待变异的个体，至少包含 'code' 字段。
            mode (str)

        返回：
            new_individual (dict): 新生成的变异个体，包含字段：
            - 'code'         : LLM 生成的 Python 代码
        """
        # 初始化个体
        new_individual = {'code': None}

        # 构建 LLM Prompt
        system, user = self.get_mutation_prompt(individual, mode)

        # 调试模式下打印 Prompt
        if self.debug_mode:
            print("\n >>> check prompt for mutation: \n", system, user)
            print(">>> Press 'Enter' to continue")
            input()

        # 调用 LLM 生成变异代码
        code = self._get_alg(system, user)

        # 调试模式下打印生成代码
        if self.debug_mode:
            print("\n >>> check mutated code: \n", code)
            print(">>> Press 'Enter' to continue")
            input()

        # 保存代码到新个体
        new_individual['code'] = code

        return new_individual

    def get_init_prompt_with(self, task):
        """
        构造生成初始个体的 Prompt。

        返回：
            tuple: (system, user)，分别为系统提示和用户提示
        """
        # 系统角色提示
        system = (
            "[System]: You are an esteemed expert in operations research and heuristic algorithm design. "
            "Your mission is to invent a novel heuristic algorithm tailored to the given optimization problem.\n"
        )

        # 函数描述
        func = (
            f"[Function I/O]: You should implement a Python function in the format of: \n"
            f"{self.prompt_func_signature} \n"
            f"The function should accept {len(self.prompt_func_inputs)} inputs: {self.joined_inputs}, "
            f"and should return {len(self.prompt_func_outputs)} output: {self.joined_outputs}.\n"
            f"{self.prompt_inout_inf}"
            f"Example Code:\n {self.prompt_func_examples}"
        )

        # 补充说明
        other = f"[Requirements]: {self.prompt_other_inf}"

        return system, task + func + other

    def get_task_description(self, pop_size):
        tasks = []
        items = self.diverse_initialization_prompt  # 更清晰的变量名
        index = 0  # 用来遍历items

        while len(tasks) < pop_size:
            tasks.append(items[index])
            index = (index + 1) % len(items)  # 当items用完时，重新从头开始
        return tasks

    def new_individual_of_task(self, task):
        """
        individual (dict): 包含至少以下字段的个体：
            - 'code'         : LLM 生成的 Python 代码字符串
        """
        individual = {'code': None}

        system, user = self.get_init_prompt_with(task)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm: \n", system, user)
            print(">>> Press 'Enter' to continue")
            input()

        code = self._get_alg(system, user, temp=1.3)

        if self.debug_mode:
            print("\n >>> check designed code: \n", code)
            print(">>> Press 'Enter' to continue")
            input()

        individual['code'] = code

        return individual


