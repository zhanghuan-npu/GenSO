import math

import hdbscan
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from eoh.src.eoh.problems.asp.replication import simulate
from eoh.src.eoh.problems.asp.clinic_environments import ClinicEnvironment

class ClusterTest:
    def __init__(self):
        self.rules = ["ibfi", "twobeg", "offset", "dome", "rule7"]
        self.cases = ClinicEnvironment()

    def ibfi(self, env_params):
        m, cv, pn, pw, cr = env_params
        n = round(210.0 / m)
        plan = [0.0] * n
        for i in range(n):
            plan[i] = i * m
        return plan

    def twobeg(self, env_params):
        m, cv, pn, pw, cr = env_params
        n = round(210.0 / m)
        plan = [0.0] * n
        for i in range(2, n):
            plan[i] = (i - 1)* m
        return plan

    def offset(self, env_params):
        m, cv, pn, pw, cr = env_params
        n = round(210.0 / m)
        plan = []
        if n == 10:
            k = 4
            for i in range(n):
                if i <= k:
                    plan.append(max(0.0, i * m + 0.15 * (i - k) * cv))
                else:
                    plan.append(i * m + 0.3 * (i - k) * cv)
        else:
            k = 8
            for i in range(n):
                if i <= k:
                    plan.append(max(0.0, i * m + 0.15 * (i - k) * cv))
                else:
                    plan.append(i * m + 0.3 * (i - k) * cv)

        return plan

    def dome(self, env_params):
        m, cv, pn, pw, cr = env_params
        n = round(210.0 / m)
        plan = [0.0] * n
        k1 = 4
        k2 = 8
        for i in range(n):
            if i <= k1:
                plan[i] = max(0.0, i * m + 0.15 * (i - k1) * m * cv)
            elif k1 < i <= k2:
                plan[i] = i * m + 0.3 * (i - k1) * m * cv
            else:
                plan[i] = i * m - 0.05 * (i - k2) * m * cv
        return plan

    def rule7(self, env_params):
        m, cv, pn, pw, cr = env_params
        n = round(210.0 / m)
        plan = [0.0] * n
        plan[0] = 0.0
        plan[1] = 0.0

        for i in range(2, n):
            plan[i] = plan[i - 1] + m + 0.3 * m * cv
            if plan[i] > 210:
                plan[i] = plan[i - 1]
        return plan

    def uar(self, env_params):
        m, cv, pn, pw, cr = env_params
        n = round(210.0 / m)
        plan = [0.0] * n
        prev = 0.0
        for i in range(1, n):
            if i == 0:
                plan[i] = 0
                prev = 0.0
            else:
                plan[i] = prev + m + (1.0 / n - 1.0 / i) * math.sqrt(cr) * m * cv
                prev = plan[i]
        return plan

    def run(self, rule, env_params):
        if rule == "ibfi":
            return self.ibfi(env_params)
        elif rule == "twobeg":
            return self.twobeg(env_params)
        elif rule == "offset":
            return self.offset(env_params)
        elif rule == "rule7":
            return self.rule7(env_params)
        elif rule == "dome":
            return self.dome(env_params)
        else:
            return self.uar(env_params)

    def evaluate(self):
        np.random.seed(42)
        G = {}
        for r, rule in enumerate(self.rules):
            print(f"正在评估规则 {rule}")
            results = []
            for c in range(self.cases.case_num):
                plan = self.run(rule, self.cases.get_case(c))
                wait, idle, over, obj = simulate(plan, self.cases.get_case(c))
                benchmark = self.cases.get_benchmark(c)
                rcf = (benchmark - obj) / benchmark
                results.append(rcf)
            G[rule] = results
        return G

    def cluster(self):
        np.random.seed(42)
        G = np.zeros((len(self.rules), self.cases.case_num))
        for r, rule in enumerate(self.rules):
            print(f"正在评估规则 {rule}")
            for c in range(self.cases.case_num):
                plan = self.run(rule, self.cases.get_case(c))
                wait, idle, over, obj = simulate(plan, self.cases.get_case(c))
                benchmark = self.cases.get_benchmark(c)
                rcf = (benchmark - obj) / benchmark
                G[r, c] = rcf  # 填充到矩阵

        for c in range(G.shape[1]):
            col = G[:, c]
            min_val = col.min()
            max_val = col.max()
            if max_val > min_val:
                G[:, c] = (col - min_val) / (max_val - min_val)
            else:
                G[:, c] = 1  # 如果该列所有值相等，则设为 0

        print("\n=== 规则-案例矩阵 G ===")
        print(G)

        X = G.T  # shape = (case_num, num_rules)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2)  # 可调参数
        labels = clusterer.fit_predict(X)
        max_label = labels.max() if len(labels) > 0 else -1
        for i, lbl in enumerate(labels):
            if lbl == -1:
                max_label += 1
                labels[i] = max_label

        clusters = {}
        for case_idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(case_idx)

        print("\n=== 聚类结果（无噪声） ===")
        for label, case_indices in clusters.items():
            print(f"子类 {label + 1} 报告案例: {', '.join(str(self.cases.get_case(i)) for i in case_indices)}")

            # 计算每条规则在该簇的平均表现
            avg_perf = G[:, case_indices].mean(axis=1)  # 对该簇列求平均
            print("规则在该簇上的平均表现:")
            for r_idx, rule in enumerate(self.rules):
                print(f"  {rule}: {avg_perf[r_idx]:.4f}")
            print()  # 空行分隔簇
        return G, labels





if __name__ == "__main__":
    ct = ClusterTest()
    ct.cluster()
