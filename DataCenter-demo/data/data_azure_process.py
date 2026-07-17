import os
import numpy as np
import pandas as pd


def process_azure_data():
    # 路径定义
    raw_path = 'raw/azure_vm_cpu_mem.csv'
    output_path = 'processed/d_hat_azur.npy'

    # 1. 加载数据
    print("Loading data...")
    df = pd.read_csv(raw_path)
    df = df.sort_values('timestamp')

    # 2. 时间聚合 (300s -> 1800s / 30min)
    # 每 6 行聚合一次 (1800 / 300 = 6)
    cpu_series = df['cpu_usage'].values
    n_aggregate = 6
    # 截断多余数据以便整除
    cpu_aggregated = cpu_series[:(len(cpu_series) // n_aggregate) * n_aggregate]
    cpu_aggregated = cpu_aggregated.reshape(-1, n_aggregate).mean(axis=1)

    # 3. 归一化到 [0, 100]
    c_min, c_max = cpu_aggregated.min(), cpu_aggregated.max()
    cpu_normalized = (cpu_aggregated - c_min) / (c_max - c_min) * 100

    # 4. 空间映射 (J=10)
    # 将长序列切分为10段，模拟10个位置
    J = 10
    min_length_per_j = len(cpu_normalized) // J
    data_j = []
    for j in range(J):
        segment = cpu_normalized[j * min_length_per_j: (j + 1) * min_length_per_j]
        data_j.append(segment)

    # 5. 块自助法采样 (Block-Bootstrap)
    S = 500
    T = 16

    # 计算每个位置可以切出多少个长度为 T 的块
    n_blocks = min_length_per_j // T
    blocks = []  # 形状: (n_blocks, J, T)

    for b in range(n_blocks):
        block = []
        for j in range(J):
            block.append(data_j[j][b * T: (b + 1) * T])
        blocks.append(block)

    blocks = np.array(blocks)  # (N, J, T)

    # 随机有放回抽样 S 次
    print(f"Sampling {S} scenarios from {len(blocks)} available blocks...")
    rng = np.random.default_rng(seed=42)  # 固定随机种子以保证结果可复现
    sample_indices = rng.choice(len(blocks), size=S, replace=True)

    d_hat_azur = blocks[sample_indices]  # (S, J, T)

    # 6. 保存结果
    if not os.path.exists('processed'):
        os.makedirs('processed')

    np.save(output_path, d_hat_azur.astype(np.float32))
    print(f"Success! Shape: {d_hat_azur.shape}")
    print(f"Max value: {d_hat_azur.max():.2f}, Min value: {d_hat_azur.min():.2f}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    process_azure_data()