import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from data.data_loader import Parameters
from math import sin, pi
import os
import json
from pathlib import Path
from evaluator_saa import Evaluator

class SAA:
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

        model = gp.Model("SAA")
        model.Params.OutputFlag = 0
        model.Params.Threads = 16
        model.Params.MIPGap = 0.01
        model.Params.Heuristics = 0.5
        model.Params.MIPFocus = 2
        model.Params.Cuts = 2

        # 第一阶段决策
        x = model.addVars(I, vtype=GRB.BINARY, name="x")
        u = model.addVars(I, lb=0, ub=U, name="u")
        a = model.addVars(I, J, vtype=GRB.BINARY, name="a")

        # 第二阶段决策
        y0 = model.addVars(I, J, T, lb=0, name="y0")

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
                        rhs -= y0[i, j, t]

                    model.addConstr(l[s, j, t] == rhs)

                #当 t == T时
                rhs = l[s, j, T - 2]
                rhs += D_s[j, T - 1]
                for i in range(I):
                    rhs -= y0[i, j, T-1]

                model.addConstr(o[s, j] == rhs)

        # 4. 关于u的约束
        for i in range(I):
            for t in range (T):
                model.addConstr(gp.quicksum(y0[i, j, t] for j in range(J)) <= u[i])

        # 5. 关于y的约束
        for i in range(I):
            for j in range(J):
                for t in range(T):
                    model.addConstr(y0[i, j, t] >= 0)
                    model.addConstr(y0[i, j, t] <= U * a[i, j])

        model.optimize()

        if model.Status == GRB.OPTIMAL:
            # 整理 solution 字典
            solution = {
                "obj": model.ObjVal,
                "x": {i: x[i].X for i in range(I)},
                "u": {i: u[i].X for i in range(I)},
                "a": {(i, j): a[i, j].X for i in range(I) for j in range(J)},
                "y0": {(i, j, t): y0[i, j, t].X for i in range(I) for j in range(J) for t in range(T)},
            }
            return solution
        else:
            print("No optimal solution found.")
            return None

def save_to_excel(data, model_name, eval_type, test_type, s, results_dir):
    file_name = "{}_{}_{}_s={}.xlsx".format(model_name, eval_type, test_type, s)
    file_path = results_dir / file_name
    df = pd.DataFrame(data, columns=["cost"])
    df.to_excel(file_path, index=False)


def save_to_json(data, model_name, s, results_dir):
    file_name = "{}_s={}.json".format(model_name, s)
    file_path = results_dir / file_name

    def tuple_to_nested_dict(data: dict):
        nested = {}
        for k, v in data.items():
            if isinstance(k, tuple):
                d = nested
                for subkey in k[:-1]:
                    d = d.setdefault(subkey, {})
                d[k[-1]] = tuple_to_nested_dict(v) if isinstance(v, dict) else v
            else:
                nested[k] = tuple_to_nested_dict(v) if isinstance(v, dict) else v
        return nested

    data_nested = tuple_to_nested_dict(data)

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data_nested, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    p = Parameters()
    # 通过参数控制两阶段or多阶段
    p.S = 100
    p.T = 2

    model_name = "saa"
    test_type = "azur"

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    solution_path = results_dir / f"{model_name}_s={p.S}.json"


    def load_saa_solution(file_path):
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return {
            "obj": data["obj"],
            "x": {int(i): v for i, v in data["x"].items()},
            "u": {int(i): v for i, v in data["u"].items()},
            "a": {
                (int(i), int(j)): v
                for i, dic_i in data["a"].items()
                for j, v in dic_i.items()
            },
            "y0": {
                (int(i), int(j), int(t)): v
                for i, dic_i in data["y0"].items()
                for j, dic_j in dic_i.items()
                for t, v in dic_j.items()
            },
        }


    # 若解不存在，则先求解并保存
    if not solution_path.exists():
        solution = SAA.solve(p)

        if solution is None:
            raise RuntimeError("SAA did not find an optimal solution.")

        save_to_json(
            data=solution,
            model_name=model_name,
            s=p.S,
            results_dir=results_dir
        )

    # 读取解
    solution = load_saa_solution(solution_path)

    # DRE 评估
    dre_res = Evaluator.decision_rule_evaluation(solution, p, test_type)

    save_to_excel(
        data=dre_res,
        model_name=model_name,
        eval_type="dre",
        test_type=test_type,
        s=p.S,
        results_dir=results_dir
    )

    print(f"Finished evaluation: {model_name}, T={p.T}, S={p.S}")


