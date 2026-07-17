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

        # Decision rule: y_t = y0 + sum_h y_h * phi
        feature_dimensions = {}
        for t in range(T):
            feature_dimensions[t] = len(SAA.get_features(t, np.ones(24)))

        y0 = model.addVars(T, lb=0, name="y0")
        y = {}
        for t in range(T):
            for h in range(feature_dimensions[t]):
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
                features = SAA.get_features(t, d_s)
                for h in range(feature_dimensions[t]):  # past demands for affine rule
                    expr_y += y[t, h] * features[h]

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
            for h in range(feature_dimensions[t]):
                coef_t.append(y[t, h].X)
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
            features = SAA.get_features(t, d)
            y_t = y_coef[t][0] + np.dot(y_coef[t][1:], features)
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
        test_data = pd.read_csv(f"instance/instance_{i}.csv").values[:, :parameters.get("T", 24)]
        solution = SAA.solve(train_data, parameters)
        cost_list = []
        for d0 in test_data:
            cost_list.append(SAA.decision_rule_evaluation(parameters, solution, d0))
        cost_array = np.array(cost_list)
        return cost_array

    @staticmethod
    def get_features(t: int, d: np.ndarray):
        features = []
        if t == 0:
            return features

        x_t_minus_1 = d[t - 1]
        features.append(x_t_minus_1)

        lookback_s = min(5, t)
        window_s = d[t - lookback_s:t]
        mean_s = np.mean(window_s)
        std_s = np.std(window_s)
        features.append(mean_s)
        features.append(std_s + 1e-8)

        lookback_m = min(20, t)
        window_m = d[t - lookback_m:t]
        mean_m = np.mean(window_m)
        features.append(x_t_minus_1 - mean_m)
        features.append((x_t_minus_1 - mean_m) / (np.std(window_m) + 1e-8))

        if t >= 2:
            features.append(d[t - 1] - d[t - 2])
        else:
            features.append(0.0)

        if t >= 2:
            vol = np.std(np.diff(d[max(0, t - 6):t])) + 1e-8
            features.append(vol)
        else:
            features.append(1e-8)

        if t == 1:
            ewm = d[0]
        else:
            alpha = 0.2
            ewm = d[t - 2]
            for i in range(t - 3, -1, -1):
                ewm = alpha * d[i] + (1 - alpha) * ewm
        features.append(ewm)

        features.append(np.sin(2 * np.pi * t / 24.0))

        cumsum = np.sum(d[:t]) / (t + 1e-8)
        features.append(cumsum)

        return features


