import numpy as np
import matplotlib.pyplot as plt

from PlotFunc import PlotBaseFunc

class BFlist:
    '''Список всевозможных базовых функций'''
    def __init__(self):
        pass

    def F1(self,x,m):
        ''' Функция x^m*(1-x) на интервале [0,1] '''
        return x**m*(1-x)

    def F2(self,x,m):
        ''' Функция x*(1-x)**m в интервале [0,1] '''
        return x*(1-x)**m
    
    def F3(self,x,y,m,n):

        return lambda x,y:self.F1(x,m)*self.F2(y,n)

class GenFromBF:
    '''Сгенерировать функцию на основе базовых'''
    def __init__(self, FuncList:list, MultList:list):

        self.FL=FuncList
        self.ML=MultList
   

    def FSum1D(self):
        
        sum_func = sum(self.ML[i]*self.FL[i] for i in range(len(self.FL)))
        
        return sum_func
        



def main():

    L=BFlist()
    x=np.linspace(0,1, num=100)
    f1=L.F1(x,2)
    f2=L.F2(x,2)
    p1=PlotBaseFunc()
    
    p1.Plot1D(x,f1)
    p1.PlotList1D([x,x],[f1,f2])
    Param=[3,2,4]
    Mult=[1,-1,1]
    sum=GenFromBF([L.F1(x,m) for m in Param],Mult).FSum1D()
   
    p1.Plot1D(x,sum)
    
    y=np.linspace(0,1, num=100)
    
    f3=L.F3(x,y,2,3)
   
    p1.PlotSurface(x,y,f3)

    pass

if __name__=='__main__':
    main()