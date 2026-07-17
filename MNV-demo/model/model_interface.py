import importlib
import numpy as np


class ModelInterface:
    """
    模型统一接口类，支持动态调用不同文件的 out_of_sample_test 函数。
    """
    # 定义文件名字符串与对应模块/类的映射关系
    # 格式: "文件名": "类名"
    MODEL_MAPPING = {
        "lp": "LP",
        "saa": "SAA",
        "saa_fh": "SAA",
        "saa_rh": "SAA",
        "saa_temp": "SAA",
        "ro": "RO",
        "dro_wass": "DRO",
        "dro_mv": "DRO"
    }

    @staticmethod
    def run_test(model_name, i, parameters):
        if model_name not in ModelInterface.MODEL_MAPPING:
            raise ValueError(f"未知的模型名称: {model_name}。可选范围: {list(ModelInterface.MODEL_MAPPING.keys())}")

        try:
            # 1. 动态导入模块 (假设 model_interface.py 与这些文件在同一目录下)
            # 使用相对导入或绝对导入，取决于你的包结构。这里使用绝对路径导入示例：
            module = importlib.import_module(f"model.{model_name}")

            # 2. 从模块中获取对应的类
            class_name = ModelInterface.MODEL_MAPPING[model_name]
            model_class = getattr(module, class_name)

            # 3. 调用静态方法并返回结果
            print(f"正在使用模型 [{model_name}] 运行算例 {i}...")
            return model_class.out_of_sample_test(i, parameters)

        except ImportError as e:
            print(f"错误: {e}")
            raise
        except AttributeError as e:
            print(f"错误: {e}")
            raise