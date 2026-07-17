"""
功能说明：
该脚本用于为鲁棒库存生产问题生成训练集、验证集和测试集需求数据，采用自回归需求生成模型。

需求生成模型：
    d_t = d*_t + Σ_{k=1}^{lag} φ_k * (d_{t-k} - d*_{t-k}) + ε_t
    ε_t ~ N(0, σ^2)

其中：
    lag    : 历史依赖期数（1~4）
    φ_k    : 内置权重 [0.4, 0.2, 0.1, 0.05] 的前 lag 个
    d*_t   : 名义需求水平
    ε_t    : 随机噪音，标准差为 σ

实验设定：
    T = 24 (规划周期)
    t = 1,...,24

名义需求函数：
    d*_t = 1000 * (1 + 0.5*sin(pi*(t-1)/12))

数据集生成规则：
    训练集，验证集，测试集
    lag = {1, 2, 3, 4}（根据实验选择）
    σ = {50}（噪音水平）
    样本数 = 50，100，1000

CSV格式要求：
    - 表头为 1~24
    - 每一行表示一个需求样本（对应 T 个时期的需求）
"""
import os
import numpy as np
import pandas as pd

# ===============================
# 基础参数设置
# ===============================

lag_list = [1, 2, 3, 4]

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

def generate_ar_demand(T = 24, lag = 4, sigma = 10, seed=None):
    # 内置权重
    # 设定总权重为 0.95，保证系统平稳但具有极强惯性
    total_phi = 0.95
    # 平分权重：
    phi = [total_phi / lag] * lag

    # 漂移项/截距：保证需求在没有噪音时能维持在一个量级（类似 1000）
    # 在 AR 模型中，长期均值 E[d] = intercept / (1 - sum(phi))
    intercept = 1000 * (1 - sum(phi))

    rng = np.random.default_rng(seed)
    d = np.zeros(T)

    # 1. 初始化：给第一期一个起始点
    d[0] = 1000 + rng.normal(0, sigma)

    # 2. 迭代生成：d_t = intercept + phi1*d_{t-1} + ... + noise
    for t in range(1, T):
        # 计算历史依赖项
        history_effect = 0
        for k in range(min(t, lag)):
            history_effect += phi[k] * d[t - (k + 1)]

        # 如果还没到 lag 期，对缺少的历史进行补位处理（保持期望一致）
        if t < lag:
            remaining_weight = sum(phi[t:])
            history_effect += remaining_weight * 1000

        noise = rng.normal(0, sigma)
        d[t] = intercept + history_effect + noise

    return np.maximum(d, 0)  # 物理约束：需求不能为负

def generate_demands(N=100, lag=4, seed=1, T=24, sigma=50):
    rng = np.random.default_rng(seed)
    return np.array([generate_ar_demand(T=T, lag=lag, sigma=sigma, seed=rng.integers(1e9)) for _ in range(N)])

# ===============================
# 保存为CSV
# ===============================

def save_csv(data, path):
    """
    保存为CSV文件
    表头为 1~24
    """
    df = pd.DataFrame(data, columns=np.arange(1, 25))
    df.to_csv(path, index=False)


# ===============================
# 生成数据集
# ===============================
for i, lag in enumerate(lag_list, start=5):
    train_data = generate_demands(N=50, lag = lag, seed=1 + i)
    train_path = os.path.join(data_dir, f"train_{i}.csv")
    save_csv(train_data, train_path)

    validate_data = generate_demands(N=100, lag = lag, seed=10 + i)
    validate_path = os.path.join(data_dir, f"validate_{i}.csv")
    save_csv(validate_data, validate_path)

    test_data = generate_demands(N=1000, lag = lag, seed=100 + i)
    test_path = os.path.join(instance_dir, f"instance_{i}.csv")
    save_csv(test_data, test_path)

print("数据生成完成。")