import json
import pandas as pd
from pathlib import Path
import time
from datetime import datetime
from data.data_loader import Parameters
from model.model_interface import ModelInterface
from model.evaluator_dro import Evaluator

def run_experiments():
    # 基础配置
    group_list = [(2, 5)] # (T, S)
    model_list = ["dro_mv"]
    data_list = ["test", "azur"]

    # 确保结果文件夹存在
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    for (t, s) in group_list:

        # 1. 初始化当前场景数下的参数
        params = Parameters(S=s)
        params.T = t

        for model_name in model_list:

            solution_path = results_dir / f"{model_name}_t={t}_s={s}.json"
            if solution_path.exists():
                print(f"Found existing solution: {solution_path}")
                solution = load_from_json(solution_path)
            else:
                print(f"Failed to load existing solution. Re-solving: {solution_path}")
                solution = ModelInterface.solve(model_name, params)
                save_to_json(data=solution, model_name=model_name, t=t, s=s, results_dir=results_dir)

            for data_name in data_list:
                # 评估方式 2: Decision Rule Evaluation (DRE)
                dre_costs = Evaluator.decision_rule_evaluation(solution, params, data_name)

                # 保存 DRE 结果
                save_to_excel(
                    data=dre_costs,
                    model_name=model_name,
                    eval_type="dre",
                    data_type=data_name,
                    t=t,
                    s=s,
                    results_dir=results_dir
                )


def save_to_excel(data, model_name, eval_type, data_type, t, s, results_dir):
    file_name = "{}_{}_{}_t={}_s={}.xlsx".format(model_name, eval_type, data_type, t, s)
    file_path = results_dir / file_name
    df = pd.DataFrame(data, columns=["cost"])
    df.to_excel(file_path, index=False)


def save_to_json(data, model_name, t, s, results_dir):
    file_name = "{}_t={}_s={}.json".format(model_name, t, s)
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

def load_from_json(file_path):
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    solution = {
        "obj": data["obj"],
        "x": {int(i): v for i, v in data["x"].items()},
        "u": {int(i): v for i, v in data["u"].items()},
        "a": {},
        "y0": {},
        "y": {}
    }

    for i, dict_j in data["a"].items():
        for j, v in dict_j.items():
            solution["a"][(int(i), int(j))] = v

    for i, dict_j in data["y0"].items():
        for j, dict_t in dict_j.items():
            for t, v in dict_t.items():
                solution["y0"][(int(i), int(j), int(t))] = v

    for i, dict_j in data["y"].items():
        for j, dict_t in dict_j.items():
            for t, dict_h in dict_t.items():
                for h, v in dict_h.items():
                    solution["y"][(int(i), int(j), int(t), int(h))] = v

    return solution


if __name__ == "__main__":
    run_experiments()
