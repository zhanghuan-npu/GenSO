import json
import os
import random
from .utils import createFolders
from .methods import methods
from .problems import problems

# 框架的调度入口类
class EVOL:

    # 初始化 EVOL 对象
    def __init__(self, paras, prob=None, **kwargs):

        print("-----------------------------------------")
        print("---              Start EoH            ---")
        print("-----------------------------------------")
        # 保存传入的参数对象
        self.paras = paras
        print("-  parameters loaded -")
        # 保存传入的问题对象
        self.prob = prob
        # 设置随机数种子
        random.seed(2024)

        
    # run methods
    def run(self, job):
        # 在 paras.exp_output_path 指定的目录下创建保存实验结果的文件夹 #
        createFolders.create_folders(self.paras.exp_output_path, job)
        print("- output folder created -")
        # 实例化一个问题生成器 Probs，并传入实验参数 paras
        problemGenerator = problems.Probs(self.paras)
        # 调用 get_problem()，实际返回一个问题对象
        problem = problemGenerator.get_problem()
        # 实例化一个方法生成器 Methods，传入实验参数和问题对象
        methodGenerator = methods.Methods(self.paras,problem)
        # 根据 paras.method 选择具体的优化方法
        method = methodGenerator.get_method()
        # 执行所选择优化方法的 run() 方法，正式开始进化优化过程
        method.run(job)

    # 多次运行函数
    def run_multiple(self, n_runs):
        summary_file = os.path.join(self.paras.exp_output_path, "summary_results.txt")
        with open(summary_file, 'w') as summary:
            summary.write("Job\tBest_Objective\n")

        for job in range(1, n_runs+1):
            print(f"Running Job {job} / {n_runs}")
            self.run(job)

            # 获取当前运行最优个体的目标值
            best_file = os.path.join(self.paras.exp_output_path, f"results_{job}", "pops_best",
                                     f"population_generation_{self.paras.ec_n_pop}.json")
            best_obj = None
            if os.path.exists(best_file):
                with open(best_file, 'r') as f:
                    best_individual = json.load(f)
                    if isinstance(best_individual, dict) and 'objective' in best_individual:
                        best_obj = best_individual['objective']

            # 写入汇总文件
            with open(summary_file, 'a') as summary:
                summary.write(f"{job}\t{best_obj}\n")

            print(f"Job {job} finished, best objective: {best_obj}")

        print(f"All {n_runs} jobs finished!")
        print("-----------------------------------------")
        print("---     EoH successfully finished !   ---")
        print("-----------------------------------------")