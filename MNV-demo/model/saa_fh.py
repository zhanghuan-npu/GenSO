import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

class SAA:
    @staticmethod
    def solve(train_data, parameters):
        # 参数
        T = parameters.get("T", 24)
        ch = parameters.get("ch", 1)
        cs = parameters.get("cs", 5)
        v0 = parameters.get("v0", 0)

        # 训练样本数量
        S = train_data.shape[0]

        model = gp.Model("SAA_inventory")
        model.Params.OutputFlag = 0

        # Decision rule: y_t = y0 + sum_h y_h * d_{t-1}
        y0 = model.addVars(T, lb=0, name="y0")
        y = {}
        for t in range(T):
            for h in range(t):
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
                # Compute affine replenishment y_t for scenario s
                expr_y = y0[t]
                for h in range(t):  # past demands for affine rule
                    expr_y += y[t, h] * d_s[h]

                # Inventory balance
                if t == 0:
                    model.addConstr(v[s, t] - xi_plus[s, t] == v0 + expr_y - d_s[t], name=f"inv_{s}_{t}")
                else:
                    model.addConstr(v[s, t] - xi_plus[s, t]  == v[s, t - 1] + expr_y - d_s[t], name=f"inv_{s}_{t}")

                model.addConstr(xi_minus[s, t] >= v[s, t], name=f"holding_{s}_{t}")

                # Replenishment bounds
                model.addConstr(expr_y >= 0, name=f"y_lb_{s}_{t}")

        # --- Objective ---
        recourse_cost = (1 / S) * gp.quicksum(cs * xi_plus[s, t] + ch * xi_minus[s, t] for s in range(S) for t in range(T))

        model.setObjective(recourse_cost, GRB.MINIMIZE)

        model.optimize()

        # --- Extract solution ---
        y_coef = []
        for t in range(T):
            coef_t = [y0[t].X]  # 截距
            # 加入已有的过去系数
            for h in range(T):
                if h < t:
                    coef_t.append(y[t, h].X)
                else:
                    coef_t.append(0.0)  # 不足的部分补0
            y_coef.append(coef_t)

        return {"y_coef": y_coef, "obj": model.ObjVal}

    @staticmethod
    def decision_rule_evaluation(parameters, solution, d):
        T = parameters.get("T", 24)
        ch = parameters.get("ch", 1)
        cs = parameters.get("cs", 10)
        v0 = parameters.get("v0", 0)
        y_coef = solution["y_coef"]

        v_list = np.zeros(T)

        total_cost = 0

        for t in range(T):
            y_t = y_coef[t][0] + np.dot(y_coef[t][1:], d)
            y_t = max(0, y_t)

            inv_prev = v0 if t == 0 else v_list[t - 1]
            v_t = inv_prev + y_t - d[t]
            v_t = max(0, v_t)
            v_list[t] = v_t

            # Holding and shortage cost
            holding = max(v_t, 0)
            shortage = max(d[t] - (inv_prev + y_t), 0)
            total_cost += ch * holding + cs * shortage

        return total_cost

    @staticmethod
    def out_of_sample_test(i, parameters):
        train_data = pd.read_csv(f"data/train_{i}.csv").values[:, :parameters.get("T", 24)]
        vali_data = pd.read_csv(f"data/validate_{i}.csv").values[:50, :parameters.get("T", 24)]
        train_data = np.vstack([train_data, vali_data])
        test_data = pd.read_csv(f"instance/instance_{i}.csv").values[:, :parameters.get("T", 24)]
        solution = SAA.solve(train_data, parameters)
        cost_list = []
        for d0 in test_data:
            cost_list.append(SAA.decision_rule_evaluation(parameters, solution, d0))
        cost_array = np.array(cost_list)
        return cost_array

