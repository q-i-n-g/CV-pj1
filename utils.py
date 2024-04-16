import numpy as np


class Relu:
    def __init__(self):
        pass

    def forward(self, x):
        return x * (x > 0)

    def backward(self, x):
        return np.ones(x.shape) * (x >= 0)


class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x):
        x = 1 / (1 + np.exp(-x))
        return x * (1 - x)


class Tanh:
    def __init__(self):
        pass

    def forward(self, x):
        return np.tanh(x)

    def backward(self, x):
        x = np.tanh(x)
        return 1 - x**2


class Softmax:
    def __init__(self):
        pass

    def forward(self, x):
        x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # 减去x中最大的值，防止指数溢出
        return x / np.sum(x, axis=-1, keepdims=True)

    def backward(self, x):
        x = x[0]
        x = self.forward(x)
        m = np.diag(x)
        for i in range(len(m)):
            for j in range(len(m)):
                if i == j:
                    m[i][j] = x[i] * (1 - x[i])
                else:
                    m[i][j] = -x[i] * x[j]
        return m


class Activation:
    def __init__(self, act_type):
        self.type = act_type

    def forward(self, x):
        if self.type == 'relu':
            return Relu().forward(x)
        if self.type == 'sigmoid':
            return Sigmoid().forward(x)
        if self.type == 'tanh':
            return Tanh().forward(x)
        if self.type == 'softmax':
            return Softmax().forward(x)
        if self.type == 'None':
            return x

    def backward(self, x):
        if self.type == 'relu':
            return Relu().backward(x)
        if self.type == 'sigmoid':
            return Sigmoid().backward(x)
        if self.type == 'tanh':
            return Tanh().backward(x)
        if self.type == 'softmax':
            return Softmax().backward(x)
        if self.type == 'None':
            return np.ones(x.shape)


class Linear:
    def __init__(self, in_features, out_features):
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.weights = np.random.randn(in_features, out_features) * scale #0.01# * scale
        self.bias = np.zeros((1, out_features))

    def forward(self, x):
        x = np.dot(x, self.weights) + self.bias
        return x


class CrossEntropyLossWithL2:
    def __init__(self, lambda_=0):
        self.lambda_ = lambda_

    def forward(self, pred, label, weight):
        epsilon = 1e-6
        pred = np.clip(pred, epsilon, 1)
        l, _ = pred.shape
        loss = -np.sum(label * np.log(pred)) / l
        l2 = 0
        for i in range(len(weight)):
            l2 += 0.5*np.sum(weight[i]**2)*self.lambda_
        return loss + self.lambda_*l2

    def backward(self, pred, label):
        epsilon = 1e-6
        pred = np.clip(pred, epsilon, 1)
        return -label / pred


class SoftmaxCrossEntropyLossWithL2:
    def __init__(self, lambda_=0):
        self.lambda_ = lambda_

    def forward(self, pred, label, weights):
        pred = Softmax().forward(pred)
        loss = CrossEntropyLossWithL2(self.lambda_).forward(pred, label, weights )
        return loss

    def backward(self, pred, label):
        l, _ = pred.shape
        return (pred - label)/l


def lr_decay(lr, iterations, weight_decay, start_up, min_lr):
    if iterations <= start_up or lr <= min_lr:
        return lr
    if iterations >= start_up and min_lr < lr:
        return lr * weight_decay


def accuracy(pred, label):
    acc = 0
    for i in range(len(pred)):
        if pred[i] == np.argmax(label[i]):
            acc += 1
    return acc/len(pred)

