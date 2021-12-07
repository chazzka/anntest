import numpy as np


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


class NN:
    def __init__(self, i, nhl, y):
        """
        :param i: all inputs
        :param nhl: eurons in hidden layer
        :param y: all outputs mapped to inputs
        """
        self.input = i
        self.y = y
        self.output = np.zeros(self.y.shape)
        # weights default 1
        self.w1 = np.ones((len(i[0]), nhl))  # 3x4 matrix
        self.w2 = np.ones((nhl, 1))  # 4x1 matrix

    def feedforward(self):
        self.layer1 = sigmoid(self.input @ self.w1)  # 4x4 matrix
        self.output = sigmoid(self.layer1 @ self.w2)  # 4x1 matrix

    def backprop(self):
        derivative1 = sigmoid_derivative(self.output)
        error = 2 * (self.y - self.output)
        derivative2 = sigmoid_derivative(self.layer1)

        d_weights2 = self.layer1.T @ (error * derivative1)  # 4x1 matrix

        d_weights1 = self.input.T @ (error * derivative1 @ self.w2.T * derivative2)

        self.w1 += d_weights1
        self.w2 += d_weights2


if __name__ == '__main__':
    # with bias appended
    inputs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])
    results = np.array([[0], [1], [1], [0]])

    nn = NN(inputs, nhl=4, y=results)
    for _ in range(10000):
        nn.feedforward()
        nn.backprop()

    print("finální output po učení")
    print(nn.output)
