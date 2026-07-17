import numpy as np
import pandas as pd
from rsome import dro
from rsome import E
from rsome import square
from rsome import norm
from rsome import grb_solver as grb
import gurobipy as gp
from gurobipy import GRB
from data.data_loader import Parameters
#from evaluator_dro import Evaluator
from pathlib import Path

"""
theta_wass 调参结果
===== T=2, S=20 =====
theta=1000, mean_cost=2739.720716, std_cost=572.155605
theta=100, mean_cost=2739.720716, std_cost=572.155605
theta=10, mean_cost=2779.405314, std_cost=600.190946
theta=1, mean_cost=2903.055801, std_cost=611.082924
theta=0.1, mean_cost=2690.645710, std_cost=654.806360
theta=0.01, mean_cost=2690.235594, std_cost=655.406105
theta=0.001, mean_cost=2754.436353, std_cost=639.203121
theta=0.0001, mean_cost=2690.015813, std_cost=655.506119
Best theta=0.0001, best_mean_cost=2690.015813
"""

class DRO:
    @staticmethod
    def solve(p: Parameters):
        # 基本参数
        I, J, T, U, S = p.I, p.J, p.T, p.U_max, p.S
        # 第一阶段成本
        f, cu, ca = p.f, p.cu, p.ca
        # 第二阶段成本
        cl, co = p.cl, p.co
        # 场景
        d_hat = p.d_hat[:S, :J, :T]  # (S, J, T)
        # 将需求矩阵压缩为不确定向量z
        z_hat = d_hat.reshape(S, J * T)

        mu = z_hat.mean(axis=0)
        sigma2 = z_hat.var(axis=0)

        z_lb = z_hat.min(axis=0)
        z_ub = z_hat.max(axis=0)

        model = dro.Model(S)

        z = model.rvar(J * T)
        uz = model.rvar()
        print(z_hat.std(axis=0).mean())

        theta_wass = 0.1

        fset = model.ambiguity()
        for s in range(S):
            z_s = z_hat[s]
            fset.suppset(z >= z_lb, z<= z_ub, norm(z - z_s, 1) <= uz)
        fset.exptset(E(uz) <= theta_wass)

        # 第一阶段决策
        x = model.dvar(I, vtype="B")
        u = model.dvar(I)
        a = model.dvar((I, J), vtype="B")

        # 仿射决策规则参数
        y = model.dvar((I, J, T))
        for t in range(1, T):
            idx = [j*T + (t-1) for j in range(J)]
            y[:, :, t].adapt(z[idx])

        # 辅助变量
        l = model.dvar((J, T-1))
        o = model.dvar(J)
        for j in range(J):
            for t in range(T - 1):
                idx = [j * T + tau for tau in range(t + 1)]
                l[j, t].adapt(z[idx])
        o.adapt(z)

        # 约束
        for j in range(J):
            model.st(a[:, j].sum() == 1)

        for i in range(I):
            for j in range(J):
                model.st(a[i, j] <= x[i])

        # 3. 关于l和o的约束
        for j in range(J):
            for t in range(T - 1):
                model.st(l[j, t] == z[j*T: j*T + (t+1)].sum() - y[:, j, :t+1].sum())
                model.st(l[j, t] >= 0)

            model.st(o[j] == z[j*T: (j+1)*T].sum() - y[:, j, :].sum())
            model.st(o[j] >= 0)

        # 4. 关于u的约束
        for i in range(I):
            model.st(u[i] <= U)
            for t in range (T):
                model.st(y[i, :, t].sum() <= u[i])


        # 5. 关于y的约束
        for i in range(I):
            for j in range(J):
                for t in range(T):
                    model.st(y[i, j, t] >= 0)
                    model.st(y[i, j, t] <= U * a[i, j])

        # 目标函数
        first_stage_cost =  f@x + cu@u + (ca*a).sum()
        recourse_cost = (cl*l).sum() + (co*o).sum()
        model.minsup(first_stage_cost + E(recourse_cost), fset)

        model.solve(grb, params={'Threads': 8, 'MIPGap': 0.01, 'Heuristics': 0.5, 'MIPFocus': 2, 'Cuts': 2})

        solution = {
                "obj": model.get(),
                "x": {i: x.get()[i] for i in range(I)},
                "u": {i: u.get()[i] for i in range(I)},
                "a": {(i, j): a.get()[i, j] for i in range(I) for j in range(J)},
                "y0": {(i, j, t): y.get()[i, j, t] for i in range(I) for j in range(J) for t in range(T)},
                "y": {
                    (i, j, t, h): y.get(z[[jp*T + (t-1) for jp in range(J)]])[i, j, t, h]
                    for i in range(I) for j in range(J) for t in range(T) for h in range(J)},
        }

        return solution

"""
if __name__ == "__main__":
    theta_wass_list = [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
    group_list = [(8, 5)]

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    log_path = results_dir / "theta_selection_log.txt"

    for group in group_list:
        p = Parameters()
        p.T = group[0]
        p.S = group[1]

        best_theta = None
        best_mean_cost = float("inf")
        best_costs = None

        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n===== T={p.T}, S={p.S} =====\n")

            for theta in theta_wass_list:
                print(f"Running theta={theta}, T={p.T}, S={p.S}")

                solution = DRO.solve(p, theta)

                if solution is None:
                    msg = f"theta={theta}: no feasible solution\n"
                    print(msg.strip())
                    f.write(msg)
                    continue

                costs = Evaluator.decision_rule_evaluation(solution, p, "test")
                mean_cost = np.mean(costs)
                std_cost = np.std(costs)

                msg = (
                    f"theta={theta}, "
                    f"mean_cost={mean_cost:.6f}, "
                    f"std_cost={std_cost:.6f}\n"
                )

                print(msg.strip())
                f.write(msg)

                if mean_cost < best_mean_cost:
                    best_mean_cost = mean_cost
                    best_theta = theta
                    best_costs = costs

            summary = (
                f"Best theta={best_theta}, "
                f"best_mean_cost={best_mean_cost:.6f}\n"
            )

            print(summary.strip())
            f.write(summary)
"""









