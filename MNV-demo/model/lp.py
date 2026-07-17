import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

class LP:
    @staticmethod
    def get_cost_of(parameters, d):
        T = parameters.get("T", 24)
        ch = parameters.get("ch", 1)
        cs = parameters.get("cs", 5)
        v0 = parameters.get("v0", 0)

        model = gp.Model("oracle_recourse")
        model.Params.OutputFlag = 0  # silent

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

        model.setObjective(gp.quicksum(cs * xi_plus[t] + ch * xi_minus[t] for t in range(T)),GRB.MINIMIZE)

        # Solve
        model.optimize()

        return model.ObjVal

    @staticmethod
    def solve(i, parameters):
        train_data = pd.read_csv(f"../instance/instance_{i}.csv").values
        cost_list = []

        for d in train_data:
            cost = LP.get_cost_of(parameters, d)
            cost_list.append(cost)

        cost_array = np.array(cost_list)
        return cost_array

    @staticmethod
    def out_of_sample_test(i, parameters, eva):
        test_data = pd.read_csv(f"instance/instance_{i}.csv").values
        cost_list = []

        for d in test_data:
            cost = LP.get_cost_of(parameters, d)
            cost_list.append(cost)

        cost_array = np.array(cost_list)
        return cost_array

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