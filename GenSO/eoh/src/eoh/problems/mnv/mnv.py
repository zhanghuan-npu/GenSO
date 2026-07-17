import concurrent
import concurrent.futures
import numpy as np
import sys
import types
import warnings
from datetime import datetime
from .saa import SAA
from .training_case import TrainingCases
from .prompts_season import GetPrompts

class MNV:
    def __init__(self):
        self.exp_settings = TrainingCases()
        self.prompts = GetPrompts()

        self.prefix = (
            "import numpy as np\n"
            "import pandas as pd\n"
            "import math\n"
            "from math import sin, cos, pi, exp, log, sqrt\n"
            "\n"
        )

    def get_case_fitness(self, alg):
        """
        根据给定的训练案例和线性决策规则，生成生产方案并评估其表现
        输出训练集的标量适应度与案例适应度
        """
        case_fit, scalar_fit  = SAA.get_case_fitness(alg, self.exp_settings.parameters, self.exp_settings.demand, self.exp_settings.val)

        return scalar_fit, case_fit

    def evaluate(self, code_string: str) -> float:
        """
        动态加载一个算法（代码字符串），并评估它在训练案例下的表现。
        算法代码必须定义所需的规则函数/类。

        评估流程：
            1. 将传入的 code_string 动态编译到一个新模块。
            2. 调用 get_case_fitness 计算适应度。
            3. 返回在训练集上面的 mean_cost 作为 fitness。

        若执行过程中出现错误，会将错误的代码保存到文件，并返回 1e9 作为惩罚性适应度。
        """
        def run_code():
            # 新建一个模块对象
            heuristic_module = types.ModuleType("heuristic_module")
            # 在新模块的命名空间里执行 code_string
            exec(self.prefix + code_string, heuristic_module.__dict__)
            # 注册模块到 sys.modules，方便 import
            sys.modules[heuristic_module.__name__] = heuristic_module

            fitness, case_fitness = self.get_case_fitness(heuristic_module)
            relative_case_fitness = []
            for i in range(self.exp_settings.case_num):
                benchmark = self.exp_settings.get_benchmark(i)
                rcf = (benchmark - case_fitness[i]) / benchmark
                relative_case_fitness.append(rcf)

            return fitness  # EOH最小化适应度

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # 使用线程池执行，并设置超时 5 分钟
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(run_code)
                    result = future.result(timeout=300)  # 300 秒 = 5 分钟
                    return result

        except concurrent.futures.TimeoutError:
            print("- Error: Execution timed out (over 5 minutes).")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"timeout_code_{timestamp}.py"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(code_string)
            return 1e9  # 超时惩罚

        except Exception as e:
            print("- Error:", str(e))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"failed_code_{timestamp}.py"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(code_string)
            return 1e9  # 其他错误惩罚

    def evaluate_ind(self, individual: dict) -> dict:
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

        def run_code():
            # 新建一个模块对象
            heuristic_module = types.ModuleType("heuristic_module")
            # 在新模块的命名空间里执行 code_string
            exec(self.prefix + code_string, heuristic_module.__dict__)
            # 注册模块到 sys.modules，方便 import
            sys.modules[heuristic_module.__name__] = heuristic_module

            case_fitness = self.get_case_fitness(heuristic_module)
            relative_case_fitness = []
            for i in range(self.exp_settings.case_num):
                benchmark = self.exp_settings.get_benchmark(i)
                rcf = (benchmark - case_fitness[i]) / benchmark
                relative_case_fitness.append(rcf)

            cf = np.asarray(relative_case_fitness, dtype=float)

            mean_cf = cf.mean()
            std_cf = cf.std(ddof=0)

            epsilon = 1e-8  # 防止除零
            avg_relative_fitness = mean_cf / (std_cf + epsilon)

            return -avg_relative_fitness  # EOH最小化适应度

        try:
            # 屏蔽警告信息
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # 使用线程池执行，并设置超时 5 分钟
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(run_code)
                    result = future.result(timeout=300)  # 300 秒 = 5 分钟

            # 更新个体
            individual["exec_success"] = True
            individual["traceback_msg"] = ""
            individual["objective"] = result
            return individual

        except concurrent.futures.TimeoutError:
            print("- Error: Execution timed out (over 5 minutes).")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"timeout_code_{timestamp}.py"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(code_string)

            # 更新个体
            individual["exec_success"] = False
            individual["traceback_msg"] = "- Error: Execution timed out (over 5 minutes)."
            individual["objective"] = float(1e9)
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
            individual["objective"] = float(1e9)
            return individual

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
            SAA.pilot_run(heuristic_module, self.exp_settings.parameters, self.exp_settings.demand)

        try:
            # 屏蔽警告信息
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # 使用线程池执行，并设置超时 5 分钟
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(run_code)
                    plans = future.result(timeout=300)  # 300 秒 = 5 分钟
            # 更新个体
            individual["exec_success"] = True
            individual["traceback_msg"] = ""
            # individual["plans"] = plans
            return individual

        except concurrent.futures.TimeoutError:
            print("- Error: Execution timed out (over 5 minutes).")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"timeout_code_{timestamp}.py"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(code_string)

            # 更新个体
            individual["exec_success"] = False
            individual["traceback_msg"] = "- Error: Execution timed out (over 5 minutes)."
            # individual["plans"] = None
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

        def run_code():
            # 新建一个模块对象
            heuristic_module = types.ModuleType("heuristic_module")
            # 在新模块的命名空间里执行 code_string
            exec(self.prefix + code_string, heuristic_module.__dict__)
            # 注册模块到 sys.modules，方便 import
            sys.modules[heuristic_module.__name__] = heuristic_module

            fitness, case_fitness = self.get_case_fitness(heuristic_module)
            relative_case_fitness = []
            for i in range(self.exp_settings.case_num):
                benchmark = self.exp_settings.get_benchmark(i)
                # 相对适应度：
                rcf = benchmark / case_fitness[i]
                relative_case_fitness.append(rcf)

            return relative_case_fitness, -fitness  # LPSE最大化适应度

        try:
            # 屏蔽警告信息
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # 使用线程池执行，并设置超时 5 分钟
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(run_code)
                    relative_case_fitness, avg_relative_fitness = future.result(timeout=300)  # 300 秒 = 5 分钟

            # 更新个体
            individual["exec_success"] = True
            individual["traceback_msg"] = ""
            individual["case_fitness"] = relative_case_fitness
            individual["objective"] = avg_relative_fitness

            return individual

        except concurrent.futures.TimeoutError:
            print("- Error: Execution timed out (over 5 minutes).")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"timeout_code_{timestamp}.py"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(code_string)

            # 更新个体
            individual["exec_success"] = False
            individual["traceback_msg"] = "- Error: Execution timed out (over 5 minutes)."
            individual["case_fitness"] = [-1e9] * self.exp_settings.case_num
            individual["objective"] = float(-1e9)
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
            individual["case_fitness"] = [-1e9] * self.exp_settings.case_num
            individual["objective"] = float(-1e9)
            return individual



