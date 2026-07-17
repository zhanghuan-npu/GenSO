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
        y = model.addVars(T, lb=0, name="y")

        # Inventory variables
        v = model.addVars(S, T, lb=0, name="v")

        # Shortage and holding
        xi_plus = model.addVars(S, T, lb=0, name="xi_plus")
        xi_minus = model.addVars(S, T, lb=0, name="xi_minus")

        # --- Constraints ---
        for s in range(S):
            d_s = train_data[s]

            for t in range(T):
                # Inventory balance
                if t == 0:
                    model.addConstr(v[s, t] - xi_plus[s, t] == v0 + y[t] - d_s[t], name=f"inv_{s}_{t}")
                else:
                    model.addConstr(v[s, t] - xi_plus[s, t] == v[s, t - 1] + y[t] - d_s[t], name=f"inv_{s}_{t}")

                model.addConstr(xi_minus[s, t] >= v[s, t], name=f"holding_{s}_{t}")

                # Replenishment bounds
                model.addConstr(y[t] >= 0, name=f"y_lb_{s}_{t}")

        # --- Objective ---
        recourse_cost = (1 / S) * gp.quicksum(cs * xi_plus[s, t] + ch * xi_minus[s, t] for s in range(S) for t in range(T))

        model.setObjective(recourse_cost, GRB.MINIMIZE)

        model.optimize()

        # --- Extract solution ---
        y0 = []
        for t in range(T):
            y0.append(y[t].X)

        return {"y": y0, "obj": model.ObjVal}

    @staticmethod
    def decision_rule_evaluation(parameters, solution, d):
        T = parameters.get("T", 24)
        ch = parameters.get("ch", 1)
        cs = parameters.get("cs", 10)
        v0 = parameters.get("v0", 0)
        y = solution["y"]

        v_list = np.zeros(T)

        total_cost = 0

        for t in range(T):
            y_t = y[t]
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

