import numpy as np
import random
import matplotlib.pyplot as plt


class Perceptron:
    """Simple perceptron"""
    weights: list
    rate: float

    def __init__(self, input_size):
        self.input_size = input_size
        self.weights = [0] * input_size
        self.rate = 0.5
        self.bias_weight = 0

    @staticmethod
    def sign(x):
        if x > 0:
            return 1
        else:
            return 0

    @staticmethod
    def relu(x):
        if x < 0:
            return 0
        else:
            return x

    def train(self, inp, desired):
        # predict what perceptron thinks now
        y = self.predict(inp)
        # update weights
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + self.rate * (desired - y) * inp[i]

        # update bias weight (inp is always 1)
        self.bias_weight = self.bias_weight + self.rate * (desired - y) * 1

        return y

    def predict(self, inp):
        res = np.dot(inp, self.weights) + 1 * self.bias_weight
        return self.sign(res)


per = Perceptron(2)
#
datasetAbove = []
datasetUnder = []

# nejaka data rozptylena
for _ in range(100):
    inp = [random.uniform(-10, 10), np.random.uniform(0, 10)]
    if inp[1] > 5:
        per.train(inp, 1)
        datasetAbove.append(inp)
    if inp[1] < 5:
        per.train(inp, 0)
        datasetUnder.append(inp)

# nejaka data vic u hranice
for _ in range(10000):
    inp = [random.uniform(-10, 10), np.random.uniform(-100, 100)]
    if inp[1] > 5:
        per.train(inp, 1)
        datasetAbove.append(inp)
    if inp[1] < 5:
        per.train(inp, 0)
        datasetUnder.append(inp)

# 100 linearly spaced numbers
x = np.linspace(-1000, 1000, 1000)

# the function
b = per.weights[0]
print(f"{per.weights} + {per.bias_weight}")
# vyjadrime y z rovnice perceptronu
y = (-(per.bias_weight / per.weights[1]) / (per.bias_weight / per.weights[0])) * x + (-per.bias_weight / per.weights[1])

# graph
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['left'].set_position(('data', 0.0))
ax.spines['bottom'].set_position(('data', 0.0))
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.gca().set_aspect('equal', adjustable='box')
UP = list(zip(*datasetAbove))
DOWN = list(zip(*datasetUnder))
plt.scatter(UP[0], UP[1], c='lightblue')
plt.scatter(DOWN[0], DOWN[1], c='red')
plt.plot(x, y)
plt.show()
