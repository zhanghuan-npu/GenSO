import importlib
import numpy as np


class ModelInterface:
    """
    模型统一接口类，支持动态调用不同文件的 solve 函数。
    """
    MODEL_MAPPING = {
        "saa_rh": "SAA",
        "saa_er": "SAA",
        "dro_mv": "DRO",
        "dro_wass": "DRO",

    }

    @staticmethod
    def solve(model_name, parameters):
        if model_name not in ModelInterface.MODEL_MAPPING:
            raise ValueError(f"未知的模型名称: {model_name}。可选范围: {list(ModelInterface.MODEL_MAPPING.keys())}")

        try:
            module = importlib.import_module(f"model.{model_name}")
            class_name = ModelInterface.MODEL_MAPPING[model_name]
            model_class = getattr(module, class_name)
            return model_class.solve(parameters)

        except ImportError as e:
            print(f"{e}")
            raise
        except AttributeError as e:
            print(f"{e}")
            raise