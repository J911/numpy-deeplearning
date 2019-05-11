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

'''
result:

1 iter loss: 12.383670705490644
2 iter loss: 0.4165706189619202
3 iter loss: 0.3458731135405174
4 iter loss: 0.32576481796165563
5 iter loss: 0.30759998463548616
6 iter loss: 0.2904847140776945
7 iter loss: 0.27432370218677493
8 iter loss: 0.25906190589047395
9 iter loss: 0.2446491938668445
10 iter loss: 0.23103832240508027
[30.707646457664925, 60.14726553732328, 89.58688461698164]
'''