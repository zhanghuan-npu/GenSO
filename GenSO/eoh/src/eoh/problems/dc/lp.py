import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from .data.data_loader import Parameters

class LP:
    @staticmethod
    def solve(p: Parameters):
        # 基本参数
        I, J, T, U, S = p.I, p.J, p.T, p.U_max, p.S
        # 第一阶段成本
        f, cu, ca = p.f, p.cu, p.ca
        # 第二阶段成本
        cl, co = p.cl, p.co
        # 场景
        d_hat = p.d_hat  # (S, J, T)

        scenario_obj_list = []

        for s in range(S):
            D_s = d_hat[s]

            model = gp.Model(f"LP_scenario_{s}")
            model.Params.OutputFlag = 0

            # 第一阶段决策变量
            x = model.addVars(I, vtype=GRB.BINARY, name="x")
            u = model.addVars(I, lb=0, ub=U, name="u")
            a = model.addVars(I, J, vtype=GRB.BINARY, name="a")

            # 第二阶段决策变量
            y = model.addVars(I, J, T, lb=0, name="y")
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
            # 1. 每个需求点必须被分配到一个设施
            for j in range(J):
                model.addConstr(gp.quicksum(a[i, j] for i in range(I)) == 1)

            # 2. 分配约束：只有启用的设施可以分配
            for i in range(I):
                for j in range(J):
                    model.addConstr(a[i, j] <= x[i])

            # 3. 推迟和外包约束
            for j in range(J):
                for t in range(T - 1):
                    if t == 0:
                        model.addConstr(l[j, t] == D_s[j, t] - gp.quicksum(y[i, j, t] for i in range(I)))
                    else:
                        model.addConstr(l[j, t] == l[j, t - 1] + D_s[j, t] - gp.quicksum(y[i, j, t] for i in range(I)))

                model.addConstr(o[j] == l[j, T - 2] + D_s[j, T - 1] - gp.quicksum(y[i, j, T - 1] for i in range(I)))

            # 4. 容量约束
            for i in range(I):
                for t in range(T):
                    model.addConstr(gp.quicksum(y[i, j, t] for j in range(J)) <= u[i])

            # 5. 分配量约束
            for i in range(I):
                for j in range(J):
                    for t in range(T):
                        model.addConstr(y[i, j, t] <= U * a[i, j])
                        model.addConstr(y[i, j, t] >= 0)

            model.optimize()

            scenario_obj_list.append(model.ObjVal)

        # print(scenario_obj_list)

        return scenario_obj_list
