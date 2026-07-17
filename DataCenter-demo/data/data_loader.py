from pathlib import Path
import pandas as pd
import numpy as np
import json

class Parameters:
    def __init__(self, S=None):
        # 建立路径
        self.data_dir = Path(__file__).resolve().parent
        self.processed_dir = self.data_dir / "processed"

        # 读取基本参数
        with open(self.processed_dir / "params.json", "r") as f:
            params = json.load(f)

        self.I = params["I"]
        self.J = params["J"]
        self.T = params["T"]
        self.U_max = params["U_max"]
        if S is not None:
            self.S = S
        else:
            self.S = params["S"]

        # 读取成本参数
        # 第一阶段，建立设施成本 f (I)
        self.f = pd.read_csv(self.processed_dir / "f.csv")["f"].values
        # 第一阶段，单位能力安装成本c_u (I,)
        self.cu = pd.read_csv(self.processed_dir / "c_u.csv")["c_u"].values
        # 第一阶段，需求传输成本c_ij (I x J)
        self.ca = pd.read_csv(self.processed_dir / "c_ij.csv", index_col=0).values
        # 第二阶段，单位推迟成本cl
        self.cl = params["c_ell"]
        # 第二阶段，单位外包成本co
        self.co = params["c_o"]

        # 读取场景与测试数据
        # d_hat[s=[200], j=[10], t=[16]]
        self.d_hat = np.load(self.processed_dir / "d_hat.npy")
        # d_hat_test[s=[500], j=[10], t=[16]]
        self.d_hat_test = np.load(self.processed_dir / "d_hat_test.npy")
        # d_hat_azur[s=[500], j=[10], t=[16]]
        self.d_hat_azur = np.load(self.processed_dir / "d_hat_azur.npy")

if __name__ == "__main__":
    p = Parameters()
