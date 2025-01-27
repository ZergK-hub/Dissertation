import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

class PlotBaseFunc:
    '''Отображать функции формы для анализа'''

    def __init__(self):
        pass
    
    def PlotIso(self,x,y,f):# отображать изолинии функции F=F(x,y)

        X, Y = np.meshgrid(x, y)

        # Calculate Z values
        Z = f(X,Y)

        # Plot isolines
        fig, ax = plt.subplots(figsize=(10, 8))
        contour = ax.contour(X, Y, Z, cmap=cm.viridis, levels=100)
        ax.clabel(contour, inline=True, fontsize=8)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Isolines of Z(x,y)')
        plt.show()
        pass

    def PlotSurface(self,x,y,f): # отображать поверхность F=F(x,y)

        X, Y = np.meshgrid(x, y)

        # Calculate Z values
        Z = f(X,Y)


        # Create the surface plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5) # Add color bar

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('F(x,y)')
        ax.set_title('Surface of F(x,y)')
        plt.show()
        pass

    def GetDataFromFEM(): #читать результаты расчета (поле) из FreeFem++
        pass
    
    def Plot1D(self,x,f):
        
        plt.plot(x,f)
        plt.show()
        pass

    def PlotList1D(self,X:list,F:list):
        ''' 
        Отобразить множество функций F=[f1(x1),f2(x2),...,fn(xn)] одной переменной на одном графике \n
        X - список, где каждый элемент сам является интервалом изменения независимой переменной \n
        F - список, где каждый элемент сам явлется множеством значений зависимой переменной \n
        '''

        [plt.plot(x,f) for x in X for f in F]

        plt.show()
    
    def RunFEM(): #запускать FreeFem++ для получения решения
        pass


def main():

    pass
    
if __name__=='__main__':
    main()