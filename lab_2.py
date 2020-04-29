import matplotlib.pyplot as plt
import math
import numpy as np
class Golden:

    def __init__(self):
        self.e = 0.05
        self.a = 1.5
        self.b = 2
        self.phi = ( math.sqrt(5) - 1) / 2



    def function(self,x):
        return  x ** 4 + 8 * x ** 3 - 6 * x ** 2 - 72 * x + 90

    def graph_func(self):
        # create 700 equally spaced points between 1 and 3
        x = np.linspace(self.a - 1, 3, 700)

        # calculate the y value for each element of the x vector
        y = self.function(x)

        fig, ax = plt.subplots()
        ax.plot(x, y)
        plt.show()

    def step(self):
        x1 = self.a + (1 - self.phi) * (self.b - self.a)
        x2 = self.a +       self.phi * (self.b - self.a)

        f1 = self.function(x1)
        f2 = self.function(x2)

        while self.b - self.a > self.e:

            if f1 > f2:
                self.a = x1
                x1 = x2
                x2 = self.a + self.phi * (self.b - self.a)
                f2 = self.function(x2)
                f1 = self.function(x1)
            else:
                self.b = x2
                x2 = x1
                x1 = self.a + (1 - self.phi) * (self.b - self.a)
                f1 = self.function(x1)
                f2 = self.function(x2)

            #print(f"a = {self.a:.3f} b = {self.b:.3f} x1 = {x1:.3f}  x2  = {x2:.3f} f1 = {f1:.3f} f2 = {f2:.2f}  b-a = {self.a - self.b:.3f}")
        else:
            return (self.b + self.a) / 2


G = Golden()
print(G.step())
G.graph_func()

