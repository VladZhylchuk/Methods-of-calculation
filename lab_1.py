import matplotlib.pyplot as plt
import numpy as np


class UniformSearch:

    def __init__(self):
        self.e = 0.1
        self.a = -2
        self.b = -3
        self.L = 605
            #abs(self.function(self.a) - self.function(self.b)) /  abs(self.a - self.b)
        self.h = 2 * self.e / self.L


    @staticmethod
    def function(x):
        return x**5 - 5*x**3 + 10*x**2  - 5*x

    @staticmethod
    def function2(x):
        return 5*x**4  + 15*x**2  +  20*x  +  5

    def graph_func(self,):
        # create 700 equally spaced points between 1 and 3
        x = np.linspace(self.a - 2, self.b + 3, 1000)

        # calculate the y value for each element of the x vector
        y = self.function2(x)

        fig, ax = plt.subplots()
        ax.plot(x, y)
        plt.ylabel('Y')
        plt.xlabel('X')

        plt.show()


    def main_loop(self):

        x = [self.a + self.h / 2]
        i = 0
        func_min = self.function(x[0])
        xmin  = x[0]
        while x[-1] < self.b:
            i += 1
            x.append(x[0] + i * self.h)

            if func_min > self.function(x[i]):
                func_min = self.function(x[i])
                xmin = x[i]
        # print(f"Повного перебору number of steps = {i} \nmin y = {func_min:.2f} \nx min = {xmin:.2f}")
        return xmin,func_min
#
Com = UniformSearch()
Com.main_loop()
Com.graph_func()


#Послідовного перебору
class EvenSearch(UniformSearch):


    def main_loop(self):
        x = [self.a + self.h / 2 ]
        i = 0
        func_min = self.function(x[i])
        xmin = x[i]

        while x[-1] < self.b:

            x.append(x[i] +  self.h + ((self.function(x[i])- func_min) / self.L))
            i += 1

            ytemp = self.function(x[-1])
            if func_min > ytemp:
                xmin = x[-1]
                func_min = ytemp

        print(f"Послідовго перебору number of steps = {i} \nmin y = {func_min:.2f} \nx min = {xmin:.2f}")


Ev = EvenSearch()
print(Ev.function2(3))
Ev.main_loop()
