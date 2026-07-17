import numpy as np
import pandas as pd
from rsome import dro
from rsome import E
from rsome import square
from rsome import norm
from rsome import grb_solver as grb
import gurobipy as gp
from gurobipy import GRB

class DRO:
    @staticmethod
    def solve(train_data, parameters, theta_wass=10):
        # 参数
        T = parameters.get("T", 24)
        t = np.arange(1, T + 1)
        d0 = 1000 * (1 + 0.5 * np.sin(np.pi * (t - 1) / 12))
        ch = parameters.get("ch", 1)
        cs = parameters.get("cs", 5)
        v0 = parameters.get("v0", 0)
        theta = parameters.get("theta", 0.2)
        wass_coef = parameters.get("wass_coef", 1e-5)

        S = train_data.shape[0]

        model = dro.Model(S)

        d = model.rvar(T)  # 定义不确定变量（需求）
        u = model.rvar()
        # theta_wass = 10

        fset = model.ambiguity()
        for s in range(S):
            d_s = train_data[s]
            fset[s].suppset(
                d >= 0,
                d >= (1 - theta) * d0,
                d <= (1 + theta) * d0,

                # Wasserstein 距离约束：联合控制 d 和 phi 到第 s 个样本的距离
                norm(d - d_s, 1) <= u,
            )

        fset.exptset(E(u) <= theta_wass)

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
            xi_plus[t].adapt(d[:t + 1])
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
        parameters["theta"] = theta_list[i-1]

        # 调节wass超参数：按 oracle_recourse_evaluation 的平均测试成本选择
        # wass_coef_list = [0.01, 0.1, 1, 10, 100]
        wass_coef_list = [10]

        best_solution = None
        best_mean_cost = np.inf
        best_wass_coef = None

        for wass_coef in wass_coef_list:
            parameters_tmp = parameters.copy()
            parameters_tmp["wass_coef"] = wass_coef

            try:
                solution_tmp = DRO.solve(train_data, parameters_tmp, wass_coef)

                tmp_cost_list = []
                for d0 in vali_data:
                    res = DRO.decision_rule_evaluation(parameters_tmp, solution_tmp, d0)
                    if res is not None:
                        tmp_cost_list.append(res)
                mean_cost_tmp = np.mean(tmp_cost_list)

                if mean_cost_tmp < best_mean_cost:
                    best_mean_cost = mean_cost_tmp
                    best_solution = solution_tmp
                    best_wass_coef = wass_coef

            except Exception as e:
                print(f"wass_coef={wass_coef} failed: {e}")

        solution = best_solution
        print("best_wass_coef =", best_wass_coef)
        print("best_mean =", best_mean_cost)

        cost_list = []
        for d0 in test_data:
            cost_list.append(DRO.decision_rule_evaluation(parameters, solution, d0))
        cost_array = np.array(cost_list)
        return cost_array

