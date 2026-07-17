#! /apps/software/gurobi1100/linux64/bin/python3.11
# -*- coding: UTF-8 -*-
import json
import pandas as pd
from pathlib import Path
import time
from datetime import datetime
from data.data_loader import Parameters
from model.model_interface import ModelInterface
from model.evaluator_azur import Evaluator


def run_experiments():
    # 基础配置
    scenarios_list = [20]
    models_to_test = ["saa_rh", "saa_er"]

    # 确保结果文件夹存在
    results_dir = Path("results_azur")
    results_dir.mkdir(exist_ok=True)
    # 定义时间记录文件路径
    time_log_path = results_dir / "solve_time_log.txt"

    for s in scenarios_list:

        # 1. 初始化当前场景数下的参数

        params = Parameters(S=s)
        print(params.d_hat_azur.max())

        for model_name in models_to_test:
            solution = json.load(open(f'results_azur/{model_name}_s=20.json'))

            ore_res = Evaluator.oracle_evaluation(solution, params)

            # 保存 ORE 结果
            save_to_excel(
                data=ore_res,
                model_name=model_name,
                eval_type="ore",
                s=s,
                results_dir=results_dir
            )

            # 评估方式 2: Decision Rule Evaluation (DRE)

            dre_costs = Evaluator.decision_rule_evaluation(solution, params, model_name)

            # 保存 DRE 结果
            save_to_excel(
                data=dre_costs,
                model_name=model_name,
                eval_type="dre",
                s=s,
                results_dir=results_dir
            )


def save_to_excel(data, model_name, eval_type, s, results_dir):
    file_name = "{}_{}_s={}.xlsx".format(model_name, eval_type, s)
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
    run_experiments()
