"""
功能说明：
该脚本用于为多周期报童问题生成训练集、验证集和测试集需求数据。

需求生成模型：
    d_t ~ U[(1-θ)d*_t , (1+θ)d*_t]

其中：
    θ      : 不确定性水平
    d*_t   : 名义需求水平

实验设定：
    T = 24 (规划周期)
    t = 1,...,24

名义需求函数：
    d*_t = 1000 * (1 + 0.5*sin(pi*(t-1)/12))

数据集生成规则：
    训练集，验证集，测试集
    θ = {2.5%, 5%, 10%, 20%}
    样本数 = 50， 100， 1000

CSV格式要求：
    - 表头为 1~24
    - 每一行表示一个需求样本
"""

import os
import numpy as np
import pandas as pd

# ===============================
# 基础参数设置
# ===============================

T = 24                          # 时间周期
N_TRAIN = 50                    # 训练样本数
N_VAL = 100                     # 验证样本数
N_TEST = 1000                   # 每个测试集样本数
theta_list = [0.025, 0.05, 0.10, 0.20]   # 不确定水平


# ===============================
# 名义需求生成
# ===============================

t = np.arange(1, T + 1)

# 名义需求 d*_t
d0 = 1000 * (1 + 0.5 * np.sin(np.pi * (t - 1) / 12))

# ===============================
# 创建文件夹
# ===============================

data_dir = "data"
instance_dir = "instance"

os.makedirs(data_dir, exist_ok=True)
os.makedirs(instance_dir, exist_ok=True)


# ===============================
# 需求生成函数
# ===============================

def generate_demands(num_samples, theta, seed):
    """
    生成需求样本

    参数
    ----------
    num_samples : int
        样本数量
    theta : float
        不确定性水平
    seed : int
        随机种子

    返回
    ----------
    ndarray
        shape = (num_samples, T)
    """
    rng = np.random.default_rng(seed)

    lower = (1 - theta) * d0
    upper = (1 + theta) * d0

    samples = rng.uniform(lower, upper, size=(num_samples, T))

    return samples


# ===============================
# 保存为CSV
# ===============================

def save_csv(data, path):
    """
    保存为CSV文件
    表头为 1~24
    """
    df = pd.DataFrame(data, columns=np.arange(1, T + 1))
    df.to_csv(path, index=False)


# ===============================
# 生成数据集
# ===============================
for i, theta in enumerate(theta_list, start=1):
    train_data = generate_demands(N_TRAIN, theta, seed=1 + i)
    train_path = os.path.join(data_dir, f"train_{i}.csv")
    save_csv(train_data, train_path)

    validate_data = generate_demands(N_VAL, theta, seed=10 + i)
    validate_path = os.path.join(data_dir, f"validate_{i}.csv")
    save_csv(validate_data, validate_path)

    test_data = generate_demands(N_TEST, theta, seed=100 + i)
    test_path = os.path.join(instance_dir, f"instance_{i}.csv")
    save_csv(test_data, test_path)

print("数据生成完成。")
