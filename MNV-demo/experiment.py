import os
import numpy as np
import pandas as pd
from model.model_interface import ModelInterface


def run_experiment():
    # --- 1. 参数设置 ---
    parameters = {
        "T": 24,
        "ch": 1.0,
        "cs": 5.0,
        "v0": 0.0
    }

    # 定义实验组
    groups = {
        #"MultiPeriod": {"instances": range(1, 5), "models": ["saa", "saa_fh", "dro_mv", "dro_wass"]}
        "MultiPeriod": {"instances": range(1, 5), "models": ["saa", "saa_fh", "dro_mv"]}
    }

    # 确保结果目录存在
    result_dir = "result"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # --- 2. 运行测试并按组合保存结果 ---
    for g_name, config in groups.items():
        for model in config["models"]:

            # 创建一个 DataFrame 用来存放该组合下所有 instance 的结果
            # 每一列代表一个 instance 的 cost_array
            combination_df = pd.DataFrame()

            for i in config["instances"]:
                # 调用接口获取 cost_array
                cost_array = ModelInterface.run_test(model, i, parameters)
                # 列名为 Instance_i
                col_name = f"Instance_{i}"
                combination_df[col_name] = cost_array

            # --- 3. 保存该组合的 Excel 文件 ---
            if not combination_df.empty:
                file_name = f"{g_name}_{model}.xlsx"
                save_path = os.path.join(result_dir, file_name)

                combination_df.to_excel(save_path, index=False)
                print(f"已保存组合结果: {file_name}")

    print(f"\n所有实验运行完毕。原始结果已按组合保存至 '{result_dir}' 文件夹。")


if __name__ == "__main__":
    run_experiment()