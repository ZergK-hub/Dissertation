import numpy as np
import matplotlib.pyplot as plt
from math import tanh

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
    
    def func1(self,x, m):
        """
        Calculates the value of the function F1(x) in range [-1,1].
        """
        return ((-x + 1) / 2)**m * ((1 + x) / 2)

    def func2(self,x, m):
        """
        Calculates the value of the function F2(x) in range [-1,1].
        """
        return ((x + 1) / 2)**m * ((1 - x) / 2)
    
    def n_m(self,x, m):
        """
        Calculates the value of the function N_m(x) = (1 - x)**m.

        Args:
            x (float or np.ndarray): The input value(s).
            m (int): The exponent.

        Returns:
            float or np.ndarray: The function value(s).
        """
        return (1-x)**m
    
    def tan_h(self,x,a):
        '''
            Вычисляет значение функции f(x)=0.5*(1-tanh(x-a)*tanh(x+a))
        '''
        return 0.5*(1-np.tanh(20*(x-a))*np.tanh(20*(x+a)))
    
    def F3(self,m,n):

        return lambda x,y:self.func1(x,m)*self.func2(x,m)*self.n_m(y,n)
    
    def F(self,a,m,n):

        return lambda x,y:self.tan_h(x,a)*self.n_m(y,m)*self.func1(x,n)*self.func2(x,n)

class GenFromBF:
    '''Сгенерировать функцию на основе базовых'''
    def __init__(self, FuncList:list, MultList:list):

        self.FL=FuncList
        self.ML=MultList
   

    def FSum1D(self):
        
        sum_func = sum(self.ML[i]*self.FL[i] for i in range(len(self.FL)))
        
        return sum_func
    
    def FSum2D(self):
        
        sum_func = lambda x,y:sum(self.ML[i]*self.FL[i](x,y) for i in range(len(self.FL)))
        
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
    
    f3=L.F3(2,3)

    x=np.linspace(-1,1, num=100)

    ftanh=L.tan_h(x,0.3)

    p1.Plot1D(x,ftanh)


    p1.PlotSurface(x,y,f3)

    VF=[
        L.F3(4,12),
        L.F3(3,2),
        L.F3(5,6)
       ]

    Mult=[1,1,1]

    S=GenFromBF(VF,Mult).FSum2D()

    p1.PlotSurface(x,y,S)

    p1.PlotIso(x,y,S)

    f=L.F(0.3,12,2)

    p1.PlotSurface(x,y,f)

    p1.PlotIso(x,y,f)


    

if __name__=='__main__':
    main()