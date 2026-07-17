from pathlib import Path

import pandas as pd
import numpy as np
from .lp import LP

"""
TrainingCases 报童问题的训练数据类

本模块定义了 TrainingCases 类，用于封装报童问题的训练数据。该类通过读取并处理外部生成的需求数据，为优化模型或算法（如规则学习、LLM 决策策略等）提供标准化的训练案例、问题特征以及对应的基准成本。

功能说明
--------
1. 类初始化：
   - 无显式参数，初始化时根据给定的不确定水平 θ 自动读取对应的训练数据文件（data/train_i.csv）。
   
2. 测试场景管理：
   - get_case(self, index)：根据索引返回指定训练案例的结构化信息，包括各时期的需求值。
   - get_benchmark(self, index)：根据索引返回对应案例的基准成本值，该值来自线性规划模型在训练需求样本上的目标值。
   - get_case_characteristic(self, index)：将案例信息与基准值组织为完整的英文描述，便于作为算法输入、LLM prompt 或实验记录。描述以“各时期的需求（英文）”开头，并列出每个时期的需求值。

3. 生成逻辑：
   - 算例1-4根据名义需求通过周期函数生成，反映需求的季节性变化。
   - 算例5-8通过自回归函数生产，反映需求的历史依赖效应。

4. 数据结构：
   - 每个训练案例对应一个需求实现，总计 50 个实现。

该类主要用于实验环境中的训练数据管理，可为优化算法、规则学习方法或基于大语言模型的决策方法提供统一的数据接口与基准结果。
"""

class TrainingCases:
    case_num = 50
    data_idx = 1

    def __init__(self):
        i = self.data_idx
        data_dir = Path(__file__).resolve().parent / "data"
        self.demand = pd.read_csv(data_dir / f"train_{i}.csv").iloc[:, :2].values
        self.val = pd.read_csv(data_dir / f"validate_{i}.csv").iloc[:, :2].values
        self.parameters = {
            "T": 2,  # 计划周期
            "cq": 5.0,  # 单位容量成本
            "ch": 1.0,  # 单位持有成本
            "cs": 10.0,  # 单位缺货成本
            "v0": 0.0  # 初始库存
        }

        cost_list = LP.solve(i, self.parameters)
        # 保存基准结果
        self.benchmark = cost_list

    def get_case(self, index):
        return self.demand[index]

    def get_benchmark(self, index):
        return self.benchmark[index]

    def get_case_characteristic(self, index):
        case = self.get_case(index)
        description = "Demand for each period: "
        # 遍历 case，拼接 "demand_i=value"
        description += ", ".join([f"demand_{i+1}={v:.2f}" for i, v in enumerate(case)])
        return description


if __name__ == "__main__":
    # 示例用法
    tc = TrainingCases()  # 初始化训练数据类
    index = 1  # 选择第一个训练案例
    print(tc.get_case(index))
    print(tc.get_benchmark(index))
    print(tc.get_case_characteristic(index))
