import numpy as np
import pandas as pd
from rsome import dro
from rsome import E
from rsome import square
from rsome import grb_solver as grb
import gurobipy as gp
from gurobipy import GRB

class DRO:
    @staticmethod
    def solve(train_data, parameters):
        # 参数
        T = parameters.get("T", 24)
        t = np.arange(1, T + 1)
        d0 = 1000 * (1 + 0.5 * np.sin(np.pi * (t - 1) / 12))
        ch = parameters.get("ch", 1)
        cs = parameters.get("cs", 5)
        v0 = parameters.get("v0", 0)
        theta = parameters.get("theta", 0.2)

        # 均值与方差
        mu = train_data.mean(axis=0)
        sigma2 = train_data.var(axis=0, ddof=1)
        sigma2 = np.maximum(sigma2, 1e-6)

        model = dro.Model()

        d = model.rvar(T)  # 定义不确定变量（需求）
        u = model.rvar(T)

        fset = model.ambiguity()
        fset.suppset(d >= (1-theta)*d0, square(d - mu) <= u, d <= (1+theta)*d0)
        fset.exptset(E(d) == mu, E(u) <= sigma2)

        # --- 3. 定义决策变量 ---
        # Wait-and-see replenishment
        y = model.dvar(T)
        for t in range(1, T):
            y[t].adapt(d[:t])

        v = model.dvar(T)
        xi_plus = model.dvar(T)
        xi_minus = model.dvar(T)
        for t in range(T):
            v[t].adapt(d[:t + 1])
            xi_plus[t].adapt(d[:t+1])
            xi_minus[t].adapt(d[:t + 1])

        model.minsup(cs * E(xi_plus.sum()) + ch * E(xi_minus.sum()), fset)


        for t in range(T):
            prev_v = v0 if t == 0 else v[t - 1]
            model.st(y[t] >= 0)
            model.st(v[t] + xi_plus[t] == prev_v + y[t] - d[t])
            model.st(v[t] >= 0)
            model.st(xi_plus[t] >= 0)
            model.st(xi_minus[t] >= v[t])
            model.st(xi_minus[t] >= 0)

        # --- 5. 求解 ---
        model.solve(grb, display=False)

        # --- Extract solution ---
        # 1. 提取截距 (Shape: T,)
        intercepts = y.get()

        # 2. 提取相对于随机变量 d 的线性系数 (Shape: T, T)
        slopes = np.nan_to_num(y.get(d), nan=0.0)
        y_coef = []
        for t in range(T):
            # 构造当前周期的系数行 [y0, y_h0, y_h1, ..., y_h{T-1}]
            row = [float(intercepts[t])]
            row.extend(slopes[t].tolist())

            y_coef.append(row)

        return {"y_coef": y_coef, "obj": model.get()}

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

        theta_list = [0.025, 0.05, 0.1, 0.2]
        parameters["theta"] = theta_list[i - 1]

        solution = DRO.solve(train_data, parameters)
        cost_list = []
        for d0 in test_data:
            cost_list.append(DRO.decision_rule_evaluation(parameters, solution, d0))
        cost_array = np.array(cost_list)
        return cost_array

