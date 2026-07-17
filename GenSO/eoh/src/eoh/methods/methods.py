class Methods():
    def __init__(self,paras,problem) -> None:
        self.paras = paras      
        self.problem = problem

    def get_method(self):
        if self.paras.method == "lpbs":
            from .LPBS.lpbs import NewMethod
            return NewMethod(self.paras,self.problem,self.select,self.manage)
        else:
            print("method "+self.method+" has not been implemented!")
            exit()