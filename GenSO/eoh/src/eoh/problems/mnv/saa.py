import os
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

class SAA:
    @staticmethod
    def solve(alg, train_data, parameters):
        # 参数
        T = parameters.get("T", 24)
        cq = parameters.get("cq", 5)
        ch = parameters.get("ch", 1)
        cs = parameters.get("cs", 10)
        v0 = parameters.get("v0", 0)

        # 训练样本数量
        S = train_data.shape[0]

        model = gp.Model("SAA_inventory")
        model.Params.OutputFlag = 0

        # Here-and-now capacity
        q = model.addVar(lb=0, name="q")

        # 预提取测试维度
        feature_dims = {}
        for t in range(T):
            sample_features = alg.get_features(t, np.ones(T))
            feature_dims[t] = len(sample_features)

        # Decision rule: y_t = y0 + sum_h y_h * d_{t-1}
        y0 = model.addVars(T, lb=0, name="y0")
        y = {}
        for t in range(T):
            for h in range(feature_dims[t]):
                y[t, h] = model.addVar(lb=-GRB.INFINITY, name=f"y_{t}_{h}")

        # Inventory variables
        v = model.addVars(S, T, lb=0, name="v")

        # Shortage and holding
        xi_plus = model.addVars(S, T, lb=0, name="xi_plus")
        xi_minus = model.addVars(S, T, lb=0, name="xi_minus")

        # --- Constraints ---
        for s in range(S):
            d_s = train_data[s]

            for t in range(T):
                # 获取该样本下，当前 t 的具体特征数值
                feature_values = alg.get_features(t, d_s)
                # Compute affine replenishment y_t for scenario s
                expr_y = y0[t]
                for h, val in enumerate(feature_values):
                    expr_y += y[t, h] * val

                # Replenishment bounds
                model.addConstr(expr_y <= q, name=f"y_ub_{s}_{t}")
                model.addConstr(expr_y >= 0, name=f"y_lb_{s}_{t}")

                # Inventory balance
                if t == 0:
                    model.addConstr(v[s, t] == v0 + expr_y - d_s[t], name=f"inv_{s}_{t}")
                else:
                    model.addConstr(v[s, t] == v[s, t - 1] + expr_y - d_s[t], name=f"inv_{s}_{t}")

                # Inventory capacity
                model.addConstr(v[s, t] <= q, name=f"cap_{s}_{t}")

                # Shortage and holding linearization
                if t == 0:
                    inv_prev = v0
                else:
                    inv_prev = v[s, t - 1]

                model.addConstr(xi_plus[s, t] >= d_s[t] - inv_prev - expr_y, name=f"shortage_{s}_{t}")
                model.addConstr(xi_minus[s, t] >= inv_prev + expr_y - d_s[t], name=f"holding_{s}_{t}")

        # --- Objective ---
        recourse_cost = (1 / S) * gp.quicksum(cs * xi_plus[s, t] + ch * xi_minus[s, t] for s in range(S) for t in range(T))

        model.setObjective(cq * q + recourse_cost, GRB.MINIMIZE)

        model.optimize()

        # --- Extract solution ---
        y_coef = []
        for t in range(T):
            coef_t = []
            coef_t.append(y0[t].X)
            # 加入已有的过去系数
            for h in range(feature_dims[t]):
                coef_t.append(y[t, h].X)
            y_coef.append(coef_t)

        return {"q": q.X, "y_coef": y_coef, "obj": model.ObjVal}

    @staticmethod
    def get_features(t: int, d: np.ndarray):
        features = []

        # 示例 1: 依赖过去 4 期的原始需求 (d_{t-1}, ..., d_{t-4})
        for k in range(max(0, t - 4), t):
            features.append(d[k])

        # 示例 2: 依赖前一期的平方项 (d_{t-1}^2)
        if t > 0:
            features.append(d[t - 1] ** 2)

        # 示例 3: 依赖前一期的余弦项 (cos(d_{t-1}))
        if t > 0:
            features.append(np.cos(d[t - 1]))
        return features

    @staticmethod
    def oracle_recourse_evaluation(parameters, solution, d):
        T = parameters.get("T", 24)
        cq = parameters.get("cq", 5)
        ch = parameters.get("ch", 1)
        cs = parameters.get("cs", 10)
        q = solution["q"]
        v0 = parameters.get("v0", 0)

        model = gp.Model("oracle_recourse")
        model.Params.OutputFlag = 0  # silent

        # Decision variables: recourse orders y_t
        y = model.addVars(T, lb=0, name="y")

        # Inventory variables v_t
        v = model.addVars(T, lb=0, ub=q, name="v")

        # Constraints: inventory balance
        for t in range(T):
            if t == 0:
                model.addConstr(v[t] == v0 + y[t] - d[t], name=f"inv_{t}")
            else:
                model.addConstr(v[t] == v[t - 1] + y[t] - d[t], name=f"inv_{t}")

        xi_plus = model.addVars(T, lb=0, name="xi_plus")
        xi_minus = model.addVars(T, lb=0, name="xi_minus")
        for t in range(T):
            inv_prev = v0 if t == 0 else v[t - 1]
            model.addConstr(xi_plus[t] >= d[t] - inv_prev - y[t], name=f"short_{t}")
            model.addConstr(xi_minus[t] >= inv_prev + y[t] - d[t], name=f"hold_{t}")

        model.setObjective(gp.quicksum(cs * xi_plus[t] + ch * xi_minus[t] for t in range(T)), GRB.MINIMIZE)

        # Solve
        model.optimize()

        # Extract obj
        total_cost = cq * q + model.ObjVal

        return total_cost

    @staticmethod
    def decision_rule_evaluation(alg, parameters, solution, d):
        T = parameters.get("T", 24)
        cq = parameters.get("cq", 5)
        ch = parameters.get("ch", 1)
        cs = parameters.get("cs", 10)
        v0 = parameters.get("v0", 0)
        q = solution["q"]
        y_coef = solution["y_coef"]

        v_list = np.zeros(T)

        total_cost = cq * q  # capacity cost

        for t in range(T):
            feature_values = alg.get_features(t, d)
            y_t = y_coef[t][0] + np.dot(y_coef[t][1:], feature_values)
            y_t = max(0, min(y_t, q))

            inv_prev = v0 if t == 0 else v_list[t - 1]
            v_t = inv_prev + y_t - d[t]
            v_t = max(0, min(v_t, q))
            v_list[t] = v_t

            # Holding and shortage cost
            holding = max(v_t, 0)
            shortage = max(d[t] - (inv_prev + y_t), 0)
            total_cost += ch * holding + cs * shortage

        return total_cost

    @staticmethod
    def get_case_fitness(alg, parameters, train_data, val_data):
        solution = SAA.solve(alg, train_data, parameters)
        cost_list = []
        for d0 in train_data:
            cost_list.append(SAA.decision_rule_evaluation(alg, parameters, solution, d0))
        cost_array_1 = np.array(cost_list)

        cost_list = []
        for d0 in val_data:
            cost_list.append(SAA.decision_rule_evaluation(alg, parameters, solution, d0))
        cost_array_2 = np.array(cost_list)

        return cost_array_1, cost_array_2.mean()+2*cost_array_2.std()

    @staticmethod
    def pilot_run(alg, parameters, train_data):
        SAA.solve(alg, train_data, parameters)

if __name__ == "__main__":
    parameters = {
        "T": 24,  # 计划周期
        "cq": 5.0,  # 单位容量成本
        "ch": 1.0,  # 单位持有成本
        "cs": 10.0,  # 单位缺货成本
        "v0": 0.0  # 初始库存
    }

    i = 1
    eva = "dre"
    print(SAA.out_of_sample_test(i, parameters, eva))

