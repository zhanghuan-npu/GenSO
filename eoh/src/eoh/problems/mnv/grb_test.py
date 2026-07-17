import numpy as np
import pandas as pd
import sys
import types
import warnings
from datetime import datetime
from saa import SAA

parameters = {
        "T": 24,  # 计划周期
        "cq": 5.0,  # 单位容量成本
        "ch": 1.0,  # 单位持有成本
        "cs": 10.0,  # 单位缺货成本
        "v0": 0.0  # 初始库存
}

prefix = (
            "import numpy as np\n"
            "import pandas as pd\n"
            "\n"
        )

code_string = "def get_features(t, d):\n    features = []\n    if t > 0:\n        features.append(d[t-1])\n    return features"

heuristic_module = types.ModuleType("heuristic_module")
# 在新模块的命名空间里执行 code_string
exec(prefix + code_string, heuristic_module.__dict__)
# 注册模块到 sys.modules，方便 import
sys.modules[heuristic_module.__name__] = heuristic_module

train_data = pd.read_csv("data/train_1.csv").values
test_data = pd.read_csv("data/train_1.csv").values
case_fit, scalar_fit = SAA.get_case_fitness(heuristic_module, parameters, train_data, test_data)
print(scalar_fit)