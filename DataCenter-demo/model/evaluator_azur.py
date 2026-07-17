import gurobipy as gp
from gurobipy import GRB
import numpy as np
import importlib


class Evaluator:
    @staticmethod
    def oracle_evaluation(solution: dict, p):
        I, J, T = p.I, p.J, p.T
        cl, co = p.cl, p.co
        d_test = p.d_hat_azur  # 形状为 (S_test, J, T)
        S_test = d_test.shape[0]

        # 提取第一阶段决策结果 (Fixed here-and-now decisions)
        x_star = solution['x']
        u_star = solution['u']
        a_star = solution['a']

        # 计算第一阶段固定成本 (Investment + Dispatch)
        # 注意：这里 solution['a'] 的 key 可能是 (i, j)
        first_stage_investment = sum(p.f[i] * x_star[str(i)] + p.cu[i] * u_star[str(i)] for i in range(I))
        first_stage_dispatch = sum(p.ca[i, j] * a_star[str(i)][str(j)] for i in range(I) for j in range(J))
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
                    model.addConstr(gp.quicksum(y[i, j, t] for j in range(J)) <= u_star[str(i)])

            # 约束 2: 分配限制 (只有第一阶段分配了 a_star[i,j]=1 的路径才能传输)
            for i in range(I):
                for j in range(J):
                    for t in range(T):
                        # y 只能在已建立的分配路径上流动，最大不超过容量或一个大数
                        model.addConstr(y[i, j, t] <= p.U_max * a_star[str(i)][str(j)])

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
    def decision_rule_evaluation(solution: dict, p, model_name):
        module = importlib.import_module(f"model.{model_name}")
        model = getattr(module, "SAA")

        I, J, T = p.I, p.J, p.T
        cl, co = p.cl, p.co
        d_test = p.d_hat_azur  # (S, J, T)
        S_test = d_test.shape[0]

        # 1. 提取训练好的第一阶段决策和决策规则系数
        x_star = solution['x']
        u_star = solution['u']
        a_star = solution['a']
        y0_star = solution['y0']  # 索引为 (i, j, t)
        y_star = solution['y']  # 索引为 (i, j, t, h)

        # 2. 计算第一阶段固定成本
        first_stage_investment = sum(p.f[i] * x_star[str(i)] + p.cu[i] * u_star[str(i)] for i in range(I))
        first_stage_dispatch = sum(p.ca[i, j] * a_star[str(i)][str(j)] for i in range(I) for j in range(J))
        fixed_cost = first_stage_investment + first_stage_dispatch

        total_costs = []

        for s in range(S_test):
            # 记录当前场景的累计运营成本
            oper_cost_s = 0
            # 初始化积压 (l_j,t)
            current_backlog = np.zeros(J)

            for t in range(T):
                # A. 计算当前时间步决策规则生成的处理量 y_ijt
                # y_ijt = y0 + sum(y_h * phi)
                y_applied = np.zeros((I, J))
                for i in range(I):
                    for j in range(J):
                        # 获取测试场景下的特征向量
                        phi = model.get_features(i, j, t, p.ca, T, d_test[s])

                        # 应用决策规则公式
                        val = y0_star[str(i)][str(j)][str(t)]
                        for h, phi_val in enumerate(phi):
                            val += y_star[str(i)][str(j)][str(t)][str(h)] * phi_val

                        # 现实约束：由于 SAA 可能不保证样本外绝对可行，
                        # 模拟部署时处理量不能为负，且不能通过未分配的路径
                        y_applied[i, j] = max(0, val) * a_star[str(i)][str(j)]

                # B. 检查并强制执行物理约束（容量限制）
                # 如果决策规则给出的总量超过了安装容量 u_i，按比例削减（或视为惩罚）
                for i in range(I):
                    total_attempted = np.sum(y_applied[i, :])
                    if total_attempted > u_star[str(i)] + 1e-7:
                        scaling_factor = u_star[str(i)] / total_attempted
                        y_applied[i, :] *= scaling_factor

                # C. 更新系统状态（积压与成本计算）
                for j in range(J):
                    # 任务处理总量
                    processed_j = np.sum(y_applied[:, j])

                    # 更新积压平衡方程: l_t = l_{t-1} + d_t - y_t
                    # 注意：如果 y_applied 过大导致 backlog 为负，说明规则给出的处理量超出了实际任务量
                    # 在现实中，你不能处理不存在的任务，因此需与 0 取 max
                    new_backlog = current_backlog[j] + d_test[s, j, t] - processed_j
                    new_backlog = max(0, new_backlog)

                    if t < T - 1:
                        # 周期内延迟成本
                        oper_cost_s += cl * new_backlog
                        current_backlog[j] = new_backlog
                    else:
                        # 最后一期强制外包成本
                        oper_cost_s += co * new_backlog

            # 总成本 = 固定成本 + 该场景模拟运行的运营成本
            total_costs.append(fixed_cost + oper_cost_s)

        return total_costs