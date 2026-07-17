import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from .data.data_loader import Parameters

class SAA:
    @staticmethod
    def solve(alg, p: Parameters):
        # 基本参数
        I, J, T, U = p.I, p.J, p.T, p.U_max
        # 第一阶段成本
        f, cu, ca = p.f, p.cu, p.ca
        # 第二阶段成本
        cl, co = p.cl, p.co
        # 场景
        d_hat = p.d_hat  # (S, J, T)
        S = 20

        model = gp.Model("SAA")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = 18000
        model.Params.Threads = 10
        model.Params.MIPGap = 0.01
        model.Params.Heuristics = 0.5
        model.Params.MIPFocus = 2
        model.Params.Cuts = 2

        # 第一阶段决策
        x = model.addVars(I, vtype=GRB.BINARY, name="x")
        u = model.addVars(I, lb=0, ub=U, name="u")
        a = model.addVars(I, J, vtype=GRB.BINARY, name="a")

        # 第二阶段决策
        phi_dims = {}
        for i in range(I):
            for j in range(J):
                for t in range(T):
                    phi_dims[i, j, t] = len(alg.get_features(i, j, t, ca, T, np.ones((J, T))))

        phi = {}
        for s in range(S):
            for i in range(I):
                for j in range(J):
                    for t in range(T):
                        phi[s, i, j, t] = alg.get_features(i, j, t, ca, T, d_hat[s])

        y0 = model.addVars(I, J, T, lb=0, name="y0")
        y = {}
        for i in range(I):
            for j in range(J):
                for t in range(T):
                    for h in range(phi_dims[i, j, t]):
                        y[i, j, t, h] = model.addVar(lb=-GRB.INFINITY, name=f"y_{i}_{j}_{t}_{h}")

        # 辅助变量
        l = model.addVars(S, J, T-1, lb=0, name="l")
        o = model.addVars(S, J, lb=0, name="o")

        # 目标函数
        obj = 0
        # 设施成本
        obj += gp.quicksum(f[i] * x[i] + cu[i] * u[i] for i in range(I))
        # 分配成本
        obj += gp.quicksum(ca[i, j] * a[i, j] for i in range(I) for j in range(J))
        # 推迟成本
        obj += (1 / S) * gp.quicksum(cl * l[s, j, t]for s in range(S) for j in range(J) for t in range(T-1))
        # 外包成本
        obj += (1 / S) * gp.quicksum(co * o[s, j] for s in range(S) for j in range(J))

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
        for s in range(S):
            D_s = d_hat[s]
            for j in range(J):
                for t in range(T - 1):
                    rhs = 0 if t == 0 else l[s, j, t-1]
                    rhs += D_s[j, t]
                    for i in range(I):
                        # phi = SAA.get_features(i, j, t, ca, T, D_s)
                        expr_y = y0[i, j, t] + gp.quicksum(y[i, j, t, h] * phi[s, i, j, t][h] for h in range(phi_dims[i, j, t]))
                        rhs -= expr_y

                    model.addConstr(l[s, j, t] == rhs)

                #当 t == T时
                rhs = l[s, j, T - 2]
                rhs += D_s[j, T - 1]
                for i in range(I):
                    # phi = SAA.get_features(i, j, T-1, ca, T, D_s)
                    expr_y = y0[i, j, T-1] + gp.quicksum(y[i, j, T-1, h] * phi[s, i, j, T-1][h] for h in range(phi_dims[i, j, T-1]))
                    rhs -= expr_y

                model.addConstr(o[s, j] == rhs)

        # 4. 关于u的约束
        for s in range(S):
            D_s = d_hat[s]
            for i in range(I):
                for t in range (T):
                    lhs = 0
                    for j in range(J):
                        # phi = SAA.get_features(i, j, t, ca, T, D_s)
                        expr_y = y0[i, j, t] + gp.quicksum(y[i, j, t, h] * phi[s, i, j, t][h] for h in range(phi_dims[i, j, t]))
                        lhs += expr_y

                    model.addConstr(lhs <= u[i])

        # 5. 关于y的约束
        for s in range(S):
            D_s = d_hat[s]
            for i in range(I):
                for j in range(J):
                    for t in range(T):
                        # phi = SAA.get_features(i, j, t, ca, T, D_s)
                        expr_y = y0[i, j, t] + gp.quicksum(y[i, j, t, h] * phi[s, i, j, t][h] for h in range(phi_dims[i, j, t]))
                        model.addConstr(expr_y >= 0)
                        model.addConstr(expr_y <= U * a[i, j])

        model.optimize()

        if model.Status == GRB.OPTIMAL:
            # 整理 solution 字典
            solution = {
                "obj": model.ObjVal,
                "x": {i: x[i].X for i in range(I)},
                "u": {i: u[i].X for i in range(I)},
                "a": {(i, j): a[i, j].X for i in range(I) for j in range(J)},
                "y0": {(i, j, t): y0[i, j, t].X for i in range(I) for j in range(J) for t in range(T)},
                "y": {(i, j, t, h): y[i, j, t, h].X for (i, j, t, h) in y.keys()},
            }
            return solution

    @staticmethod
    def get_features(i, j, t, ca, T, D_s):
        """
        输入：参数i, j, t, ca(I*J), T, D_s (J*T)
        输出：特征list
        """
        if t == 0:
            # t=0 没有历史 → 设为0
            return []
        else:
            return D_s[:, t - 1]


    @staticmethod
    def tuple_to_nested_dict(data: dict):
        nested = {}
        for k, v in data.items():
            if isinstance(k, tuple):
                d = nested
                for subkey in k[:-1]:
                    d = d.setdefault(subkey, {})
                d[k[-1]] = SAA.tuple_to_nested_dict(v) if isinstance(v, dict) else v
            else:
                nested[k] = SAA.tuple_to_nested_dict(v) if isinstance(v, dict) else v
        return nested

    @staticmethod
    def pilot_run(alg, p: Parameters):
        # 基本参数
        I, J, T, U, S = 3, 5, 4, p.U_max, 1
        # 第一阶段成本
        f, cu, ca = p.f, p.cu, p.ca
        ca = ca[:I, :J]  # 形状 (3, 5)
        # 第二阶段成本
        cl, co = p.cl, p.co
        # 场景
        d_hat = p.d_hat  # (S, J, T)
        d_hat = d_hat[:S, :J, :T]  # 形状 (1, 5, 4)

        model = gp.Model("SAA")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = 5
        model.Params.Threads = 1
        model.Params.MIPGap = 0.10
        model.Params.Heuristics = 0.5
        model.Params.MIPFocus = 2
        model.Params.Cuts = 2

        # 第一阶段决策
        x = model.addVars(I, vtype=GRB.BINARY, name="x")
        u = model.addVars(I, lb=0, ub=U, name="u")
        a = model.addVars(I, J, vtype=GRB.BINARY, name="a")

        # 第二阶段决策
        phi_dims = {}
        for i in range(I):
            for j in range(J):
                for t in range(T):
                    phi_dims[i, j, t] = len(alg.get_features(i, j, t, ca, T, np.ones((J, T))))

        phi = {}
        for s in range(S):
            for i in range(I):
                for j in range(J):
                    for t in range(T):
                        phi[s, i, j, t] = alg.get_features(i, j, t, ca, T, d_hat[s])

        y0 = model.addVars(I, J, T, lb=0, name="y0")
        y = {}
        for i in range(I):
            for j in range(J):
                for t in range(T):
                    for h in range(phi_dims[i, j, t]):
                        y[i, j, t, h] = model.addVar(lb=-GRB.INFINITY, name=f"y_{i}_{j}_{t}_{h}")

        # 辅助变量
        l = model.addVars(S, J, T - 1, lb=0, name="l")
        o = model.addVars(S, J, lb=0, name="o")

        # 目标函数
        obj = 0
        # 设施成本
        obj += gp.quicksum(f[i] * x[i] + cu[i] * u[i] for i in range(I))
        # 分配成本
        obj += gp.quicksum(ca[i, j] * a[i, j] for i in range(I) for j in range(J))
        # 推迟成本
        obj += (1 / S) * gp.quicksum(cl * l[s, j, t] for s in range(S) for j in range(J) for t in range(T - 1))
        # 外包成本
        obj += (1 / S) * gp.quicksum(co * o[s, j] for s in range(S) for j in range(J))

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
        for s in range(S):
            D_s = d_hat[s]
            for j in range(J):
                for t in range(T - 1):
                    rhs = 0 if t == 0 else l[s, j, t - 1]
                    rhs += D_s[j, t]
                    for i in range(I):
                        # phi = SAA.get_features(i, j, t, ca, T, D_s)
                        expr_y = y0[i, j, t] + gp.quicksum(
                            y[i, j, t, h] * phi[s, i, j, t][h] for h in range(phi_dims[i, j, t]))
                        rhs -= expr_y

                    model.addConstr(l[s, j, t] == rhs)

                # 当 t == T时
                rhs = l[s, j, T - 2]
                rhs += D_s[j, T - 1]
                for i in range(I):
                    # phi = SAA.get_features(i, j, T-1, ca, T, D_s)
                    expr_y = y0[i, j, T - 1] + gp.quicksum(
                        y[i, j, T - 1, h] * phi[s, i, j, T - 1][h] for h in range(phi_dims[i, j, T - 1]))
                    rhs -= expr_y

                model.addConstr(o[s, j] == rhs)

        # 4. 关于u的约束
        for s in range(S):
            D_s = d_hat[s]
            for i in range(I):
                for t in range(T):
                    lhs = 0
                    for j in range(J):
                        # phi = SAA.get_features(i, j, t, ca, T, D_s)
                        expr_y = y0[i, j, t] + gp.quicksum(
                            y[i, j, t, h] * phi[s, i, j, t][h] for h in range(phi_dims[i, j, t]))
                        lhs += expr_y

                    model.addConstr(lhs <= u[i])

        # 5. 关于y的约束
        for s in range(S):
            D_s = d_hat[s]
            for i in range(I):
                for j in range(J):
                    for t in range(T):
                        # phi = SAA.get_features(i, j, t, ca, T, D_s)
                        expr_y = y0[i, j, t] + gp.quicksum(
                            y[i, j, t, h] * phi[s, i, j, t][h] for h in range(phi_dims[i, j, t]))
                        model.addConstr(expr_y >= 0)
                        model.addConstr(expr_y <= U * a[i, j])

        model.optimize()

if __name__ == "__main__":
    p = Parameters()
    SAA.solve(p)
