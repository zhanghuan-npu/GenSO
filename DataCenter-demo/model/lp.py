import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from data.data_loader import Parameters
from math import sin, pi

class LP:
    @staticmethod
    def solve(p: Parameters, dataset):
        # 基本参数
        I, J, T, U, S = p.I, p.J, p.T, p.U_max, p.S
        # 第一阶段成本
        f, cu, ca = p.f, p.cu, p.ca
        # 第二阶段成本
        cl, co = p.cl, p.co
        # 场景
        if dataset == "azur":
            d_hat = p.d_hat_azur[:S, :J, :T]  # (S, J, T)
        else:
            d_hat = p.d_hat_test[:S, :J, :T]  # (S, J, T)
        cost_list = []

        for d in d_hat:
            model = gp.Model()
            model.Params.OutputFlag = 0
            model.Params.Threads = 1

            # 第一阶段决策
            x = model.addVars(I, vtype=GRB.BINARY, name="x")
            u = model.addVars(I, lb=0, ub=U, name="u")
            a = model.addVars(I, J, vtype=GRB.BINARY, name="a")
            # 第二阶段决策
            y = model.addVars(I, J, T, lb=0, name="y")
            # 辅助变量
            l = model.addVars(J, T - 1, lb=0, name="l")
            o = model.addVars(J, lb=0, name="o")

            # 目标函数
            obj = 0
            # 设施成本
            obj += gp.quicksum(f[i] * x[i] + cu[i] * u[i] for i in range(I))
            # 分配成本
            obj += gp.quicksum(ca[i, j] * a[i, j] for i in range(I) for j in range(J))
            # 推迟成本
            obj += gp.quicksum(cl * l[j, t] for j in range(J) for t in range(T - 1))
            # 外包成本
            obj += gp.quicksum(co * o[j] for j in range(J))

            model.setObjective(obj, GRB.MINIMIZE)

            # 约束
            # 1. sum_{J} a_ij == 1
            for j in range(J):
                model.addConstr(gp.quicksum(a[i, j] for i in range(I)) == 1)
            # 2. a_ij <= x_i
            for i in range(I):
                for j in range(J):
                    model.addConstr(a[i, j] <= x[i])

            # 3. 关于l和o的约束
            for j in range(J):
                model.addConstr(l[j, 0] == d[j, 0] - gp.quicksum(y[i, j, 0] for i in range(I)))
                for t in range(1, T-1):
                    model.addConstr(l[j, t] == l[j, t-1] + d[j, t] - gp.quicksum(y[i, j, t] for i in range(I)))

                model.addConstr(o[j] == l[j, T-2] + d[j, T-1] - gp.quicksum(y[i, j, T-1] for i in range(I)))

            # 4. 关于u的约束
            for i in range(I):
                for t in range(T):
                    model.addConstr(gp.quicksum(y[i, j, t] for j in range(J)) <= u[i])

            # 5. 关于y的约束
            for i in range(I):
                for j in range(J):
                    for t in range(T):
                        model.addConstr(y[i, j, t] <= U * a[i, j])

            model.optimize()

            cost_list.append(model.ObjVal)

        return cost_list


if __name__ == "__main__":
    p = Parameters()
    p.T = 8
    p.S = 500
    dataset = "azur"
    cost_list = LP.solve(p, dataset)
    # 保存结果
    df = pd.DataFrame({"LowerBound": cost_list})
    df.to_excel("t={} {} lb.xlsx".format(p.T, dataset), index=False)