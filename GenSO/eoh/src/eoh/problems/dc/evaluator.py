import numpy as np

class Evaluator:
    @staticmethod
    def decision_rule_evaluation(alg, solution: dict, p):
        I, J, T = p.I, p.J, p.T
        cl, co = p.cl, p.co
        d_test = p.d_hat  # (S, J, T)
        S = p.S

        # 1. 提取训练好的第一阶段决策和决策规则系数
        x_star = solution['x']
        u_star = solution['u']
        a_star = solution['a']
        y0_star = solution['y0']  # 索引为 (i, j, t)
        y_star = solution['y']  # 索引为 (i, j, t, h)

        # 2. 计算第一阶段固定成本
        first_stage_investment = sum(p.f[i] * x_star[i] + p.cu[i] * u_star[i] for i in range(I))
        first_stage_dispatch = sum(p.ca[i, j] * a_star[i, j] for i in range(I) for j in range(J))
        fixed_cost = first_stage_investment + first_stage_dispatch

        total_costs = []

        for s in range(S):
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
                        phi = alg.get_features(i, j, t, p.ca, T, d_test[s])

                        # 应用决策规则公式
                        val = y0_star[(i, j, t)]
                        for h, phi_val in enumerate(phi):
                            val += y_star.get((i, j, t, h), 0) * phi_val

                        # 现实约束：由于 SAA 可能不保证样本外绝对可行，
                        # 模拟部署时处理量不能为负，且不能通过未分配的路径
                        y_applied[i, j] = max(0, val) * a_star[(i, j)]

                # B. 检查并强制执行物理约束（容量限制）
                # 如果决策规则给出的总量超过了安装容量 u_i，按比例削减（或视为惩罚）
                for i in range(I):
                    total_attempted = np.sum(y_applied[i, :])
                    if total_attempted > u_star[i] + 1e-7:
                        scaling_factor = u_star[i] / total_attempted
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