import numpy as np
import pandas as pd
from rsome import ro
from rsome import norm
from rsome import grb_solver as grb
import gurobipy as gp
from gurobipy import GRB

class RO:
    @staticmethod
    def solve(train_data, parameters):
        # 参数
        T = parameters.get("T", 24)
        t = np.arange(1, T + 1)
        d0 = 1000 * (1 + 0.5 * np.sin(np.pi * (t - 1) / 12))
        cq = parameters.get("cq", 5)
        ch = parameters.get("ch", 1)
        cs = parameters.get("cs", 10)
        v0 = parameters.get("v0", 0)
        theta = parameters.get("theta", 0.2)

        # 均值与方差
        mu = train_data.mean(axis=0)
        sigma2 = train_data.var(axis=0, ddof=1)
        sigma2 = np.maximum(sigma2, 1e-6)
        sigma = np.sqrt(sigma2)

        rho = parameters.get("rho", 2.0)

        model = ro.Model("Robust_Inventory")

        d = model.rvar(T)  # 定义不确定变量（需求）

        # d_min = train_data.min(axis=0)
        # d_max = train_data.max(axis=0)
        # uset = (d >= d_min, d <= d_max)

        # uset = (d >= (1-theta)*d0, d <= (1+theta)*d0)
        uset = (norm((d - mu) * (1 / sigma), 2) <= rho, d >= 0)

        # --- 3. 定义决策变量 ---
        # Here-and-now capacity
        q = model.dvar()

        # Wait-and-see replenishment
        y = model.ldr(T)
        for t in range(1, T):
            y[t].adapt(d[:t])

        xi = model.dvar(T)

        model.minmax(cq * q + xi.sum(), uset)


        for t in range(T):
            model.st(y[t] >= 0, y[t] <= q)
            model.st(v0 + y[:t+1].sum() - d[:t+1].sum() >= 0)
            model.st(v0 + y[:t+1].sum() - d[:t+1].sum() <= q)
            model.st(xi[t] >= cs * (d[:t+1].sum() - v0 - y[:t+1].sum()))
            model.st(xi[t] >= ch * (v0 + y[:t+1].sum() - d[:t+1].sum()))


        # --- 5. 求解 ---
        model.solve(grb, display=False)

        # --- 6. 提取解并转换格式 ---
        # 获取 q
        q_val = q.get()

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

        return {"q": q_val, "y_coef": y_coef, "obj": model.get()}

    @staticmethod
    def oracle_recourse_evaluation(parameters, solution, d):
        T = parameters.get("T", 24)
        cq = parameters.get("cq", 5)
        ch = parameters.get("ch", 1)
        cs = parameters.get("cs", 10)
        q = solution["q"]
        v0 = parameters.get("v0", 0)

        model = gp.Model("oracle_recourse")
        model.Params.OutputFlag = 0  # silent

        # Decision variables: recourse orders y_t
        y = model.addVars(T, lb=0, ub=q, name="y")

        # Inventory variables v_t
        v = model.addVars(T, lb=0, ub=q, name="v")

        # Constraints: inventory balance
        for t in range(T):
            if t == 0:
                model.addConstr(v[t] == v0 + y[t] - d[t], name=f"inv_{t}")
            else:
                model.addConstr(v[t] == v[t - 1] + y[t] - d[t], name=f"inv_{t}")

        xi_plus = model.addVars(T, lb=0, name="xi_plus")
        xi_minus = model.addVars(T, lb=0, name="xi_minus")
        for t in range(T):
            inv_prev = v0 if t == 0 else v[t - 1]
            model.addConstr(xi_plus[t] >= d[t] - inv_prev - y[t], name=f"short_{t}")
            model.addConstr(xi_minus[t] >= inv_prev + y[t] - d[t], name=f"hold_{t}")

        model.setObjective(gp.quicksum(cs * xi_plus[t] + ch * xi_minus[t] for t in range(T)), GRB.MINIMIZE)

        # Solve
        model.optimize()

        if model.status != gp.GRB.Status.OPTIMAL:
            return None

        # Extract obj
        total_cost = cq * q + model.ObjVal

        return total_cost

    @staticmethod
    def decision_rule_evaluation(parameters, solution, d):
        T = parameters.get("T", 24)
        cq = parameters.get("cq", 5)
        ch = parameters.get("ch", 1)
        cs = parameters.get("cs", 10)
        v0 = parameters.get("v0", 0)
        q = solution["q"]
        y_coef = solution["y_coef"]

        v_list = np.zeros(T)

        total_cost = cq * q  # capacity cost

        for t in range(T):
            y_t = y_coef[t][0] + np.dot(y_coef[t][1:], d)
            y_t = max(0, min(y_t, q))

            inv_prev = v0 if t == 0 else v_list[t - 1]
            v_t = inv_prev + y_t - d[t]
            v_t = max(0, min(v_t, q))
            v_list[t] = v_t

            # Holding and shortage cost
            holding = max(v_t, 0)
            shortage = max(d[t] - (inv_prev + y_t), 0)
            total_cost += ch * holding + cs * shortage

        return total_cost

    @staticmethod
    def out_of_sample_test(i, parameters, eva):
        train_data = pd.read_csv(f"data/train_{i}.csv").values[:, :parameters.get("T", 24)]
        test_data = pd.read_csv(f"instance/instance_{i}.csv").values[:, :parameters.get("T", 24)]

        theta_list = [0.025, 0.05, 0.1, 0.2]
        parameters["theta"] = theta_list[i-1]

        solution = RO.solve(train_data, parameters)
        cost_list = []
        for d0 in test_data:
            if eva == "ore":
                res = RO.oracle_recourse_evaluation(parameters, solution, d0)
                if res is not None:
                    cost_list.append(res)
            elif eva == "dre":
                cost_list.append(RO.decision_rule_evaluation(parameters, solution, d0))
        cost_array = np.array(cost_list)
        return cost_array

if __name__ == "__main__":
    parameters = {
        "T": 24,  # 计划周期
        "cq": 5.0,  # 单位容量成本
        "ch": 1.0,  # 单位持有成本
        "cs": 10.0,  # 单位缺货成本
        "v0": 0.0  # 初始库存
    }

    i = 1
    eva = "ore"
    print(RO.out_of_sample_test(i, parameters, eva))

