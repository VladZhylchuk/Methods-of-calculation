import matplotlib.pyplot as plt
import numpy as np

class Fibonacci:
    def __init__(self):
        self.a = 1.5
        self.b = 2
        self.e = 0.05
        self.condition = (self.b - self.a) / self.e

    @staticmethod
    def function(x):
        return x**4 + 8 * x**3 - 6 * x**2 - 72 * x + 90

    def graph_func(self):
        # create 700 equally spaced points between 1 and 3
        x = np.linspace(1, 3, 700)

        # calculate the y value for each element of the x vector
        y = self.function(x)

        fig, ax = plt.subplots()
        ax.plot(x, y)
        plt.show()

    @staticmethod
    def fnum(f_number):
        fib1 = fib2 = 1

        for i in range(2, f_number):
            fib1, fib2 = fib2, fib1 + fib2

        return fib2


    def find_fnum(self):
        j = 0
        while  self.fnum(j) <= self.condition:
            j += 1
        return self.fnum(j),j


    def find_x(self):

        j,i = self.find_fnum()
        fnum1 = self.fnum(i - 2)       #F(i - 2) number
        fnum2 = self.fnum(i - 1)       #F(i - 1) number
        fnum  = self.fnum(i)

        x1 = self.a + (fnum1 / fnum ) * (self.b - self.a)
        x2 = self.a + (fnum2 / fnum ) * (self.b - self.a)
        return x1,x2

    def main_loop(self):

        x1,x2 = self.find_x()
        f1 = self.function(x1)
        f2 = self.function(x2)
        j, i = self.find_fnum()
        k = 0

        while abs(self.b - self.a) > self.e :

            if f1 <= f2 :

                fnum3 = self.fnum(i - 3 - k)  # F(i - 3) number
                fnum1 = self.fnum(i - 1 - k)  # F(i - 2) number

                self.b = x2
                x2 = x1
                x1 = self.a + (fnum3 / fnum1) * (self.b - self.a)
                f1 = self.function(x1)
                f2 = self.function(x2)
                k += 1
            else:

                fnum2 = self.fnum(i - 2 - k)  # F(i - 2) number
                fnum1 = self.fnum(i - 1 - k)  # F(i - 2) number

                self.a = x1
                x1 = x2
                x2 = self.a + (fnum2 / fnum1) * (self.b - self.a)
                f1 = self.function(x1)
                f2 = self.function(x2)
                k += 1
        return (self.a + self.b) / 2

F = Fibonacci()
x = F.main_loop()

print(f"x min = {x:.2f} \ny min = {F.function(x):.2f}")
F.graph_func()
# for i in range(1,10):
#     print(F.fnum(i))

# print(F.find_fnum())

