"""
Probs 问题生成工厂类

本模块定义了 `Probs` 类，用于根据实验参数 `paras` 自动生成优化问题实例。
该设计采用工厂模式，可以根据参数动态选择不同类型的问题，或直接使用
外部传入的自定义问题对象，便于进化算法或搜索方法调用。

功能说明
--------
1. 问题来源：
   - 可以直接传入一个问题对象（非字符串），`Probs` 会直接加载该对象作为优化问题。
   - 也可以通过指定字符串名称，由 `Probs` 动态导入并实例化对应的内置问题类。

2. 内置问题：
   - "tsp_construct" → 旅行商构造问题（TSP Construct）
   - "bp_online"     → 在线背包问题（BP Online）
   - 如果传入的字符串名称不在已实现列表中，程序会提示未找到该问题并报错。

3. 方法接口：
   - `get_problem()`：
       - 返回加载完成的具体问题实例。
       - 在算法运行时，`EVOL` 或具体优化方法会通过此接口获取问题对象，
         以便进行适应度评估或算法操作。
"""

class Probs():
    def __init__(self,paras):

        if not isinstance(paras.problem, str):
            self.prob = paras.problem
            print("- Prob local loaded ")
        elif paras.problem == "mnv":
            from eoh.src.eoh.problems.mnv import mnv
            self.prob = mnv.MNV()
            print("- Prob " + paras.problem + " loaded ")
        elif paras.problem == "dc":
            from eoh.src.eoh.problems.dc import dc
            self.prob = dc.DC()
            print("- Prob " + paras.problem + " loaded ")
        else:
            print("problem "+paras.problem+" not found!")


    def get_problem(self):

        return self.prob
