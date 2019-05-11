import random
from sympy import Symbol, diff, lambdify, exp
import numpy as np


def sigmoid(x):
    result = []
    for val in x:
        result.append(1 / (1 + exp(-1*val)))
    return result

class Model():
    def __init__(self, lr):
        self._weight1 = random.random()
        self._weight2 = random.random()
        self._bias = random.random()

        self.weight1 = Symbol('weight1')
        self.weight2 = Symbol('weight2')
        self.bias = Symbol('bias')
        self.lr = lr
        
    def forward(self, x):
        x = np.array(x, dtype='float')
        hypothesis = sigmoid(x[:,0] * self.weight1 + x[:,1] * self.weight2 + self.bias)
        output = lambdify((self.weight1, self.weight2, self.bias), hypothesis)
        return output(self._weight1, self._weight2, self._bias), hypothesis
    
    def train(self, x, y):
        x, y = np.array(x, dtype='float'), np.array(y, dtype='float')
        _, hypothesis = self.forward(x)
        cost = np.square(y - hypothesis).sum() / y.size

        weight1_diff = lambdify((self.weight1, self.weight2, self.bias), diff(cost, self.weight1))
        weight2_diff = lambdify((self.weight1, self.weight2, self.bias), diff(cost, self.weight2))
        bias_diff = lambdify((self.weight1, self.weight2, self.bias), diff(cost, self.bias))

        self._weight1 = self._weight1 - self.lr * weight1_diff(self._weight1, self._weight2, self._bias)
        self._weight2 = self._weight2 - self.lr * weight2_diff(self._weight1, self._weight2, self._bias)
        self._bias = self._bias - self.lr * bias_diff(self._weight1, self._weight2, self._bias)

        loss = lambdify((self.weight1, self.weight2, self.bias), cost)
        return(loss(self._weight1, self._weight2, self._bias))


if __name__ == "__main__":
    # AND GATE example
    x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_data = [0, 0, 0, 1]

    # generate Model instance
    model = Model(lr = 0.1)

    # train 1000 iter
    for step in range(1, 1001):
        loss = model.train(x_data, y_data)
        print(step, 'iter', 'loss:', loss)

    # predict
    output, _ = model.forward(x_data)
    print(output)

    loss = model.train(x_data, y_data)
