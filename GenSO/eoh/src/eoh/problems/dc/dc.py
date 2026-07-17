import numpy as np
import sys
import types
import warnings
from datetime import datetime
from uuid import uuid4
from .saa import SAA
from .training_case import TrainingCases
from .prompts import GetPrompts
from .evaluator import Evaluator


class DC:
    def __init__(self):
        self.exp_settings = TrainingCases()
        self.prompts = GetPrompts()

        self.prefix = (
            "import numpy as np\n"
            "import pandas as pd\n"
            "import math\n"
            "from math import sin, cos, pi, exp, log, sqrt\n"
            "from typing import List\n"
            "\n"
        )

    def get_case_fitness(self, alg):
        solution = SAA.solve(alg, self.exp_settings.p)
        cost_list = Evaluator.decision_rule_evaluation(alg, solution, self.exp_settings.p)
        cost_array = np.array(cost_list)
        return solution, cost_array.mean() + 2*cost_array.std(), cost_array

    def pilot_run(self, individual):
        """
        对个体进行 Pilot Run（试运行），主要检查其代码是否能在测试用例上正常执行。

        输入:
            individual (dict): 至少包含以下字段的个体：
                - 'code' (str): LLM 生成的 Python 代码字符串
                更新字段
                - individual["exec_success"] (bool): 是否成功执行
                - individual["traceback_msg"] (str): 若失败则记录报错信息，否则为空字符串

        返回:
            individual (dict): 更新后的个体，带有执行状态信息
        """
        code_string = individual["code"]

        def run_code():
            # 新建一个模块对象
            heuristic_module = types.ModuleType("heuristic_module")
            # 在新模块的命名空间里执行 code_string
            exec(self.prefix + code_string, heuristic_module.__dict__)
            # 注册模块到 sys.modules，方便 import
            sys.modules[heuristic_module.__name__] = heuristic_module
            # 试运行
            SAA.pilot_run(heuristic_module, self.exp_settings.p)

        try:
            # 屏蔽警告信息
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                run_code()
            # 更新个体
            individual["exec_success"] = True
            individual["traceback_msg"] = ""
            # individual["plans"] = plans
            return individual

        except Exception as e:
            traceback_msg = str(e)
            print("- Error:", str(e))

            # 追加保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"failed_code_{timestamp}.py"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(code_string)

            # 更新个体
            individual["exec_success"] = False
            individual["traceback_msg"] = traceback_msg
            return individual

    def evaluate_case_fitness(self, individual: dict) -> dict:
        """
        评估一个个体（individual），更新其适应度和执行状态。

        输入:
            individual (dict): 至少包含 'code' 字段的个体

        更新字段:
            - individual["obj"] : 适应度值（float）
            - individual["exec_success"] : 是否成功执行（bool）
            - individual["traceback_msg"] : 错误信息（若失败）

        返回:
            individual (dict): 更新后的个体
        """
        code_string = individual["code"]
        module_name = f"heuristic_{uuid4().hex}"

        def run_code():
            # 新建一个模块对象
            heuristic_module = types.ModuleType(module_name)
            # 在新模块的命名空间里执行 code_string
            exec(self.prefix + code_string, heuristic_module.__dict__)
            # 注册模块到 sys.modules，方便 import
            sys.modules[heuristic_module.__name__] = heuristic_module

            solution, fitness, case_fitness = self.get_case_fitness(heuristic_module)
            solution = SAA.tuple_to_nested_dict(solution)

            relative_case_fitness = []
            for i in range(self.exp_settings.case_num):
                benchmark = self.exp_settings.get_benchmark(i)
                # 相对适应度：
                rcf = benchmark / case_fitness[i]
                relative_case_fitness.append(rcf)

            return solution, relative_case_fitness, -fitness  # LPSE最大化适应度

        try:
            solution, relative_case_fitness, avg_relative_fitness = run_code()
            individual.update({
                "exec_success": True,
                "traceback_msg": "",
                "solution": solution,
                "case_fitness": relative_case_fitness,
                "objective": avg_relative_fitness
            })

            return individual

        except Exception as e:
            traceback_msg = str(e)
            print("- Error:", str(e))

            # 追加保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"failed_code_{timestamp}.py"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(code_string)

            # 更新个体
            individual.update({
                "exec_success": False,
                "traceback_msg": str(e),
                "solution": {},
                "case_fitness": [-1e9] * self.exp_settings.case_num,
                "objective": float(-1e9),
            })
            return individual
        finally:
            if module_name in sys.modules:
                del sys.modules[module_name]
