from pylab import meshgrid, cm,  title
import numpy as np
from numpy import arange
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

class GradientMethod:

    def __init__(self):
        self.e = 0.001
        self.beta  = 0.6
        self.x = 4
        self.y = 1
        self.x2 = 1
        self.y2 = 1.2
        self.R = self.absolute((self.dfdx(self.x,self.y) - self.dfdx(self.x2,self.y2)),(self.dfdy(self.x,self.y) - self.dfdy(self.x2,self.y2))) \
                  / self.absolute((self.x - self.x2),(self.y - self.y2))
        self.alpha = (1 - self.e) / self.R


    @staticmethod
    def function(x,y):
        return 3*x**2 - 3*x*y + 4*y**2 - 2*x + y

    @staticmethod
    def function2(xy):
        return 3 * xy[0] ** 2 - 3 * xy[0] * xy[1] + 4 * xy[0] ** 2 - 2 * xy[0] + xy[1]

    def graph_func(self):

        plt.style.use('seaborn-paper')
        x = arange(-3.0, 3.0, 0.1)
        y = arange(-3.0, 3.0, 0.1)
        X, Y = meshgrid(x, y)  # grid of point
        Z = self.function(X, Y)  # evaluation of the function on the grid
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=cm.RdBu, linewidth=0, antialiased=False)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=5)
        title('$z=3x^2 - 3xy + 4y^2 - 2x + y $')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    #Похідна по х (х1)
    def dfdx(self,x,y):
        return 6*x -3*y - 2
    # Похідна по у (х2)
    def dfdy(self,x,y):
        return 8*y - 3*x + 1
    # Повертає градінтний вектор []
    def grad_vector(self,x,y):
        return  np.array([self.dfdx(x,y), self.dfdy(x,y)])

    def absolute(self,x,y):
        return (x**2 + y**2)**0.5

    def main_loop(self):
        # Перша точка з якої почнемо спускатися Х(3,1)
        X = [3,1]
        iteration = 1
        x_prev = np.array(X)
        x_next = np.subtract(x_prev, self.alpha * self.grad_vector(x_prev[0], x_prev[1]))


        # while np.linalg.norm(np.subtract(x_next,x_prev)) > self.e:
        while self.absolute(self.dfdx(x_next[0],x_next[1]), self.dfdy(x_prev[0],x_prev[1])) > self.e:

            check_null = self.grad_vector(x_prev[0], x_prev[1])
            if self.absolute(check_null[0], check_null[1]) == 0:
                return x_prev,iteration
            else:

                x_prev = x_next
                x_next = x_prev - self.alpha * self.grad_vector(x_prev[0],x_prev[1])


                a = self.function(x_prev[0] - self.alpha*self.dfdx(x_prev[0],x_prev[1]),
                                  x_prev[1] - self.alpha*self.dfdy(x_prev[0],x_prev[1])) - self.function(x_prev[0],x_prev[1])

                b = -self.e*self.alpha*abs(np.linalg.norm(np.multiply(
                     self.grad_vector(x_prev[0],x_prev[1]), self.grad_vector(x_prev[0],x_prev[1]))))


                if  a <= b:
                    iteration += 1
                else:
                    self.alpha *= self.beta
        return x_prev, iteration




# G = GradientMethod()
# arrxy,num_of_itaretion = G.main_loop()
# print(f"x min = {arrxy[0]:.3f} \ny min = {arrxy[1]:.3f} \nNumber of iteration`s = {num_of_itaretion}")
# G.graph_func()



class Steppest(GradientMethod):


    def main_loop(self):
        # Перша точка з якої почнемо спускатися Х(3,1)
        X = [3,1]
        iteration = 1
        x_prev = np.array(X)
        x_next = np.subtract(x_prev, self.alpha * self.grad_vector(x_prev[0], x_prev[1]))

        # while np.linalg.norm(np.subtract(x_next,x_prev)) > self.e:
        while self.absolute(self.dfdx(x_next[0],x_next[1]), self.dfdy(x_next[0],x_next[1])) > self.e:


            minf =  self.function2(x_prev -  self.grad_vector(x_prev[0],x_prev[1]))
            min_alpha = 0
            for alpha in range(1,200):

                if minf > self.function2(x_prev -  (alpha/100) * self.grad_vector(x_prev[0],x_prev[1])):
                    minf = self.function2(x_prev -  (alpha/100) * self.grad_vector(x_prev[0],x_prev[1]))
                    min_alpha = alpha/100
            self.alpha = min_alpha
            check_null = self.grad_vector(x_prev[0], x_prev[1])
            if self.absolute(check_null[0], check_null[1]) == 0:
                return x_prev,iteration
            else:

                x_prev = x_next
                x_next = x_prev - self.alpha * self.grad_vector(x_prev[0],x_prev[1])


                a = self.function(x_prev[0] - self.alpha*self.dfdx(x_prev[0],x_prev[1]),
                                  x_prev[1] - self.alpha*self.dfdy(x_prev[0],x_prev[1])) - self.function(x_prev[0],x_prev[1])

                b = -self.e*self.alpha*abs(np.linalg.norm(np.multiply(
                     self.grad_vector(x_prev[0],x_prev[1]), self.grad_vector(x_prev[0],x_prev[1]))))


                if  a <= b:
                    iteration += 1
                # else:
                #     self.alpha *= self.beta
        return x_prev, iteration
# S = Steppest()
# arrxy,num_of_itaretion = S.main_loop()
# print(f"x min = {arrxy[0]:.3f} \ny min = {arrxy[1]:.3f} \nNumber of iteration`s = {num_of_itaretion}")
# S.graph_func()

