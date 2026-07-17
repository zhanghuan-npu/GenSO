import os
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

class LP:
    @staticmethod
    def get_cost_of(parameters, d):
        T = parameters.get("T", 24)
        cq = parameters.get("cq", 5)
        ch = parameters.get("ch", 1)
        cs = parameters.get("cs", 10)
        v0 = parameters.get("v0", 0)

        model = gp.Model("oracle_recourse")
        model.Params.OutputFlag = 0  # silent

        q = model.addVar(lb=0, name="q")
        # Decision variables: recourse orders y_t
        y = model.addVars(T, lb=0, name="y")
        # Inventory variables v_t
        v = model.addVars(T, lb=0, name="v")
        # 辅助变量
        xi_plus = model.addVars(T, lb=0, name="xi_plus")
        xi_minus = model.addVars(T, lb=0, name="xi_minus")

        for t in range(T):
            # Constraints: inventory balance
            inv_prev = v0 if t == 0 else v[t - 1]

            # 库存平衡方程（含缺货补偿 xi_plus，确保 v[t] >= 0）
            model.addConstr(v[t] == inv_prev + y[t] - d[t] + xi_plus[t], name=f"bal_{t}")
            # 持有量 xi_minus 实际上就是期末库存 v[t]
            model.addConstr(xi_minus[t] == v[t], name=f"hold_{t}")
            # 容量约束：补货量和库存量都不能超过决策的容量 q
            model.addConstr(y[t] <= q, name=f"cap_y_{t}")
            model.addConstr(v[t] <= q, name=f"cap_v_{t}")

        model.setObjective(cq * q + gp.quicksum(cs * xi_plus[t] + ch * xi_minus[t] for t in range(T)),GRB.MINIMIZE)

        # Solve
        model.optimize()

        return model.ObjVal

    @staticmethod
    def solve(i, parameters):
        base_dir = os.path.dirname(__file__)
        data_path = os.path.join(base_dir, "data", f"train_{i}.csv")
        train_data = pd.read_csv(data_path).values
        cost_list = []

        for d in train_data:
            cost = LP.get_cost_of(parameters, d)
            cost_list.append(cost)

        return cost_list

if __name__ == '__main__':
    parameters = {
        "T": 24,  # 计划周期
        "cq": 5.0,  # 单位容量成本
        "ch": 1.0,  # 单位持有成本
        "cs": 10.0,  # 单位缺货成本
        "v0": 0.0  # 初始库存
    }

    i = 1
    print(LP.solve(i, parameters))