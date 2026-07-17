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
    def decision_rule_evaluation(parameters, solution, d, alpha):
        T = parameters.get("T", 24)
        ch = parameters.get("ch", 1)
        cs = parameters.get("cs", 10)
        v0 = parameters.get("v0", 0)
        y_coef = solution["y_coef"]

        v_list = np.zeros(T)

        total_cost = 0

        for t in range(T):
            y_t = alpha * (y_coef[t][0] + np.dot(y_coef[t][1:], d))
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
    def oda_sequential_boost(parameters, train_data):
        alpha_values = np.arange(0.8, 1.21, 0.01)

        best_alpha = None
        best_obj = float('inf')

        n_samples = train_data.shape[0]
        n_folds = 5
        fold_size = n_samples // n_folds

        for alpha in alpha_values:
            total_obj = 0

            # 10折交叉验证
            for fold in range(n_folds):
                # 计算当前折的验证集索引
                start_idx = fold * fold_size
                end_idx = (fold + 1) * fold_size if fold < n_folds - 1 else n_samples

                # 验证集：当前折的数据
                validation_set = train_data[start_idx:end_idx]

                # 训练集：除当前折外的所有数据
                train_indices = list(range(0, start_idx)) + list(range(end_idx, n_samples))
                train_set = train_data[train_indices]

                # 在训练集上求解
                solution = SAA.solve(train_set, parameters)

                # 在验证集上评估（所有验证样本的平均）
                fold_obj = 0
                for d in validation_set:
                    fold_obj += SAA.decision_rule_evaluation(parameters, solution, d, alpha)
                fold_obj = fold_obj / validation_set.shape[0]

                total_obj += fold_obj

            # 计算10折平均目标值
            avg_obj = total_obj / n_folds

            # 更新最优值
            if avg_obj < best_obj:
                best_obj = avg_obj
                best_alpha = alpha

        return best_alpha

    @staticmethod
    def out_of_sample_test(i, parameters):
        train_data = pd.read_csv(f"data/train_{i}.csv").values[:, :parameters.get("T", 24)]
        validation_data = pd.read_csv(f"data/validate_{i}.csv").values[:, :parameters.get("T", 24)]
        test_data = pd.read_csv(f"instance/instance_{i}.csv").values[:, :parameters.get("T", 24)]
        solution = SAA.solve(train_data, parameters)
        # ODA的序贯提升函数
        alpha = SAA.oda_sequential_boost(parameters, train_data)
        cost_list = []
        for d0 in test_data:
            cost_list.append(SAA.decision_rule_evaluation(parameters, solution, d0, alpha))
        cost_array = np.array(cost_list)
        return cost_array

    @staticmethod
    def oda_feature(t, d):
        """
        计算ODA风格的特征字典

        参数:
            t: 当前期数 (从1开始)
            d: np.array, 保存从t=1~T的所有需求

        返回:
            list: 包含10个ODA特征的列表，t=1时返回空列表
        """
        # 如果t=1，没有历史数据，返回空列表
        if t == 1:
            return []

        n = t - 1  # 历史数据数量

        # 计算季节性需求偏差 e_tau
        # d_tau^* = 1000 * (1 + 0.5 * sin(pi * (tau - 1) / 12))
        e = np.zeros(n)
        for tau in range(1, n + 1):
            d_star = 1000 * (1 + 0.5 * np.sin(np.pi * (tau - 1) / 12))
            e[tau - 1] = d[tau - 1] - d_star

        # 获取历史需求数据 (前n期)
        d_history = d[:n]

        # 计算基本统计量
        d_n = d_history[-1]  # 最新需求 d_n
        d_bar_n = np.mean(d_history)  # 历史均值

        # phi_3: 指数加权平均 (lambda=0.7)
        lambda_val = 0.7
        weights = np.array([lambda_val ** (n - 1 - i) for i in range(n)])
        phi_3 = np.sum(weights * d_history) / np.sum(weights)

        # phi_4: 累积季节性偏差
        phi_4 = np.sum(e)

        # phi_5: 正部 (累积季节性偏差的正部)
        phi_5 = max(phi_4, 0)

        # phi_6: 负部 (累积季节性偏差的负部)
        phi_6 = max(-phi_4, 0)

        # phi_7: 动量 (当前需求与上一期需求的差)
        if n >= 2:
            phi_7 = d_history[-1] - d_history[-2]
        else:
            phi_7 = 0

        # phi_8: 历史峰值需求
        phi_8 = np.max(d_history)

        # phi_9: 标准差 (波动性)
        phi_9 = np.sqrt(np.mean((d_history - d_bar_n) ** 2))

        # phi_10: 需求高于名义水平的频率
        # 需要重新计算每个历史期的名义需求
        d_star_history = np.array([
            1000 * (1 + 0.5 * np.sin(np.pi * (tau - 1) / 12))
            for tau in range(1, n + 1)
        ])
        phi_10 = np.mean(d_history > d_star_history)

        # 构建特征列表，按照公式中的顺序
        features = [
            d_n,  # phi_1: 最新需求
            d_bar_n,  # phi_2: 历史均值
            phi_3,  # phi_3: 指数加权平均
            phi_4,  # phi_4: 累积季节性偏差
            phi_5,  # phi_5: 正部
            phi_6,  # phi_6: 负部
            phi_7,  # phi_7: 动量
            phi_8,  # phi_8: 峰值需求
            phi_9,  # phi_9: 波动性
            phi_10  # phi_10: 超名义需求频率
        ]

        return features


