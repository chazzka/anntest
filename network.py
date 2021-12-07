import numpy as np


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
        self.w1 = np.ones((len(i[0]), nhl)).T  # 4x3 matrix

        self.w2 = np.ones((nhl, 1))  # 4x1 matrix

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1.0 - x)

    def feedforward(self):
        self.layer1 = self.sigmoid(np.dot(self.input, self.w1.T))  # 4x4 matrix
        # print("layer1")
        # print(self.layer1)
        # print("w2 = Wo")
        # print(self.w2)
        # už není třeba transpose, je to matice
        self.output = self.sigmoid(np.dot(self.layer1, self.w2))  # 4x1 matrix
        print("output - po feedforwardu")
        print(self.output)

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1

        derivative1 = self.sigmoid_derivative(self.output)

        print("sigmoid derivative")
        print(derivative1)

        error = 2 * (self.y - self.output)
        d_weights2 = self.layer1.T @ (error * derivative1)

        print("vahy outputu (zpětné procházení)")
        print(d_weights2)

        derivative2 = self.sigmoid_derivative(self.layer1)

        d_weights1 = self.input.T @ (error * derivative1 @ self.w2.T * derivative2)
        print("váhy skryté vsrtvy (zpětné procházení)")
        print(d_weights1.T)
        # update the weights with the derivative (slope) of the loss function
        self.w1 += d_weights1.T
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
