import numpy as np
import gurobipy as gp
from gurobipy import GRB



class Evaluator:
    @staticmethod
    def oracle_evaluation(solution: dict, p, test_type: str):
        I, J, T = p.I, p.J, p.T
        cl, co = p.cl, p.co
        if test_type == "test":
            d_test = p.d_hat_test  # 形状为 (S_test, J, T)
        elif test_type == "azur":
            d_test = p.d_hat_azur
        S_test = d_test.shape[0]

        # 提取第一阶段决策结果 (Fixed here-and-now decisions)
        x_star = solution['x']
        u_star = solution['u']
        a_star = solution['a']

        # 计算第一阶段固定成本 (Investment + Dispatch)
        # 注意：这里 solution['a'] 的 key 可能是 (i, j)
        first_stage_investment = sum(p.f[i] * x_star[i] + p.cu[i] * u_star[i] for i in range(I))
        first_stage_dispatch = sum(p.ca[i, j] * a_star[i, j] for i in range(I) for j in range(J))
        fixed_cost = first_stage_investment + first_stage_dispatch

        total_costs = []

        # 对每一个测试场景并行（或循环）求解最优追溯
        for s in range(S_test):
            model = gp.Model(f"Oracle_Recourse_s{s}")
            model.Params.OutputFlag = 0  # 关闭日志输出
            model.Params.Threads = 1  # 子问题较小，单线程即可

            # 第二阶段决策变量：y 不再受决策规则限制，而是 wait-and-see 决策
            # y[i, j, t] 表示在场景 s 下，t 时刻从中心 i 分配给需求点 j 的量
            y = model.addVars(I, J, T, lb=0, name="y")
            l = model.addVars(J, T - 1, lb=0, name="l")
            o = model.addVars(J, lb=0, name="o")

            # 目标函数：最小化当前场景下的运营成本
            oper_cost = gp.quicksum(cl * l[j, t] for j in range(J) for t in range(T - 1)) + gp.quicksum(
                co * o[j] for j in range(J))
            model.setObjective(oper_cost, GRB.MINIMIZE)

            # 约束 1: 容量限制 (受限于第一阶段确定的 u_star)
            for i in range(I):
                for t in range(T):
                    model.addConstr(gp.quicksum(y[i, j, t] for j in range(J)) <= u_star[i])

            # 约束 2: 分配限制 (只有第一阶段分配了 a_star[i,j]=1 的路径才能传输)
            for i in range(I):
                for j in range(J):
                    for t in range(T):
                        # y 只能在已建立的分配路径上流动，最大不超过容量或一个大数
                        model.addConstr(y[i, j, t] <= p.U_max * a_star[i, j])

            # 约束 3: 积压与流量平衡
            for j in range(J):
                for t in range(T):
                    prev_l = l[j, t - 1] if t > 0 else 0
                    arrival = d_test[s, j, t]
                    processed = gp.quicksum(y[i, j, t] for i in range(I))

                    if t < T - 1:
                        # 过程中的积压
                        model.addConstr(l[j, t] == prev_l + arrival - processed)
                    else:
                        # 最后一期的外包
                        model.addConstr(o[j] == prev_l + arrival - processed)

            # 求解
            model.optimize()

            total_costs.append(fixed_cost + model.ObjVal)
        # 计算样本外性能指标 J^(1)
        return total_costs

    @staticmethod
    def decision_rule_evaluation(solution: dict, p, test_type: str):
        I, J, T = p.I, p.J, p.T
        if test_type == "test":
            d_test = p.d_hat_test  # 形状为 (S_test, J, T)
        elif test_type == "azur":
            d_test = p.d_hat_azur
        S_test = d_test.shape[0]

        # 1. 提取训练好的第一阶段决策和决策规则系数
        x_star = solution['x']
        u_star = solution['u']
        a_star = solution['a']
        y0_star = solution['y0']  # 索引为 (i, j, t)

        # 2. 计算第一阶段固定成本
        first_stage_investment = sum(p.f[i] * x_star[i] + p.cu[i] * u_star[i] for i in range(I))
        first_stage_dispatch = sum(p.ca[i, j] * a_star[i, j] for i in range(I) for j in range(J))
        fixed_cost = first_stage_investment + first_stage_dispatch

        total_costs = []

        for s in range(S_test):
            oper_cost_s = 0
            current_backlog = np.zeros(J)

            for t in range(T):
                y_applied = np.zeros((I, J))

                for i in range(I):
                    for j in range(J):
                        y_applied[i, j] = max(0, y0_star[(i, j, t)]) * a_star[(i, j)]

                # 容量修正
                for i in range(I):
                    total_attempted = np.sum(y_applied[i, :])
                    u_i = u_star[i]
                    if total_attempted > u_i + 1e-7:
                        scaling_factor = u_star[i] / total_attempted
                        y_applied[i, :] *= scaling_factor

                # 状态更新与成本计算
                for j in range(J):
                    processed_j = np.sum(y_applied[:, j])
                    new_backlog = current_backlog[j] + d_test[s, j, t] - processed_j
                    new_backlog = max(0, new_backlog)

                    if t < T - 1:
                        oper_cost_s += p.cl * new_backlog
                        current_backlog[j] = new_backlog
                    else:
                        oper_cost_s += p.co * new_backlog

            total_costs.append(fixed_cost + oper_cost_s)

        return total_costs