import random
from sympy import Symbol, diff, lambdify
import numpy as np

class Model():
    def __init__(self, lr):
        self._weight = random.random()
        self._bias = random.random()

        self.weight = Symbol('weight')
        self.bias = Symbol('bias')
        self.lr = lr
        
    def forward(self, x):
        x = np.array(x, dtype='float')
        hypothesis = x * self.weight + self.bias
        output = lambdify((self.weight, self.bias), hypothesis)
        return output(self._weight, self._bias), hypothesis
    
    def train(self, x, y):
        x, y = np.array(x, dtype='float'), np.array(y, dtype='float')
        _, hypothesis = self.forward(x)
        cost = np.square(y - hypothesis).sum() / y.size

        weight_diff = lambdify((self.weight, self.bias), diff(cost, self.weight))
        bias_diff = lambdify((self.weight, self.bias), diff(cost, self.bias))

        self._weight = self._weight - self.lr * weight_diff(self._weight, self._bias)
        self._bias = self._bias - self.lr * bias_diff(self._weight, self._bias)

        loss = lambdify((self.weight, self.bias), cost)
        return(loss(self._weight, self._bias))


if __name__ == "__main__":
    x_data = [1, 2, 3]
    y_data = [30, 60, 90]

    # generate Model instance
    model = Model(lr = 0.1)

    # train 10 iter
    for step in range(1, 11):
        loss = model.train(x_data, y_data)
        print(step, 'iter', 'loss:', loss)

    # predict
    output, _ = model.forward(x_data)
    print(output)
