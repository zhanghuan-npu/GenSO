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
        uz = model.rvar(J * T)

        fset = model.ambiguity()
        fset.suppset(z >= 0, z <= z_ub, square(z - mu) <= uz)
        fset.exptset(E(z) == mu, E(uz) <= sigma2)

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

if __name__ == "__main__":
    p = Parameters()

    DRO.solve(p)








