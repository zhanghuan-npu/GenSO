import numpy as np
from .lp import LP
from .data.data_loader import Parameters

class TrainingCases:
    def __init__(self):
        self.p = Parameters()
        self.case_num = self.p.S
        self.case = self.p.d_hat
        # self.benchmark = LP.solve(self.p)
        self.benchmark = [1000.0]*self.case_num
    def get_case(self, index):
        return self.case[index]

    def get_benchmark(self, index):
        return self.benchmark[index]

    def get_case_characteristic(self, index):
        D_s = self.get_case(index)  # D_s.shape = (J, T)
        J, T = D_s.shape
        description = f"Scenario {index} Characteristics:\n"

        for j in range(J):
            demand = D_s[j, :]
            mean_d = np.mean(demand)
            max_d = np.max(demand)
            min_d = np.min(demand)
            std_d = np.std(demand)
            # 高峰期判定
            peaks = np.where(demand > mean_d + std_d)[0]
            peak_info = f"Peaks at periods {peaks.tolist()}" if len(peaks) > 0 else "No significant peaks"

            description += (f"  Demand point {j}: mean={mean_d:.2f}, max={max_d}, min={min_d}, "
                            f"std={std_d:.2f}, {peak_info}\n")

        return description


if __name__ == "__main__":
    # 示例用法
    tc = TrainingCases()  # 初始化训练数据类
    index = 1  # 选择第一个训练案例
    print(tc.get_case(index))
    print(tc.get_benchmark(index))
    print(tc.get_case_characteristic(index))
