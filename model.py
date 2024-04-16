import utils
import numpy as np
import json
import os
import shutil


class Model:
    def __init__(self, hidden_layers=(128, 32), activation=('relu', 'relu', 'softmax'), input_size=784,
                 output_size=10, lambda_=0.0005):
        self.config = {'hidden_layers': hidden_layers, 'activation': activation, 'lambda_': lambda_}
        self.layers = []
        for i in range(3):
            if i == 0:
                self.layers.append(utils.Linear(input_size, hidden_layers[i]))
            else:
                if i == 2:
                    self.layers.append(utils.Linear(hidden_layers[i - 1], output_size))
                else:
                    self.layers.append(utils.Linear(hidden_layers[i - 1], hidden_layers[i]))
            self.layers.append(utils.Activation(activation[i]))

        self.lambda_ = lambda_
        self.loss = utils.SoftmaxCrossEntropyLossWithL2(self.lambda_)
        self.wx = []  # fc层后结果(x1,x2,x3)   x0->x1->a1->x2->a2->x3->a3
        self.ax = []  # 激活函数层后结果, 第一个是输入(x0,a1,a2)

    def forward(self, x):
        self.refresh()
        self.ax.append(x)
        for i in range(len(self.layers)):
            x = self.layers[i].forward(x)
            if i % 2:
                self.ax.append(x)
            else:
                self.wx.append(x)
        self.ax = self.ax[:-1]
        return x

    def predict(self, x):
        x = self.forward(x)
        return np.argmax(x, axis=1)

    def backward(self, x, y):
        grad_b = [0, 0, 0]
        prob = self.forward(x)
        for i in range(len(self.layers))[-1::-2]:
            if i == len(self.layers) - 1:
                grad_b[i // 2] = self.loss.backward(prob,
                                                    y)
            else:
                grad_b[i // 2] = np.dot(grad_b[i // 2 + 1], self.layers[i + 1].weights.T) * self.layers[i].backward(
                    self.wx[i // 2])
        return grad_b, prob

    def refresh(self):  # 清空存储的用于计算导数的中间值
        self.ax = []
        self.wx = []

    def update(self, x, y, lr):
        grad_b, prob = self.backward(x, y)
        loss = self.loss.forward(self.wx[2], y, [0, 0, 0])
        for i in range(len(self.layers))[::2]:
            grad_w = np.dot(self.ax[i // 2].T, np.atleast_2d(grad_b[i // 2])) + self.lambda_ * self.layers[i].weights
            self.layers[i].weights = self.layers[i].weights - lr * grad_w
            self.layers[i].bias = self.layers[i].bias - lr * np.sum(grad_b[i // 2], axis=0)
        return loss

    def test(self, X, Y):
        pred = []
        for i in range(len(X)):
            pred.append(self.predict(X[i]))
        return utils.accuracy(pred, Y)

    def para(self):
        para = [0, 0, 0]
        for i in range(len(self.layers))[::2]:
            para[i // 2] = {}
            para[i // 2]['weights'] = self.layers[i].weights
            para[i // 2]['bias'] = self.layers[i].bias
        return para

    def train(self, X, Y, X_test, Y_test, lr=0.05, weight_decay=1, batch_size=1, epochs=1, start_up=200, min_lr=1e-4,
              save=False, output_dir='./output/test'):
        losses = []
        accs = []
        max_acc = 0
        for epoch in range(epochs):
            indices = np.random.choice(len(X), len(X), replace=True)
            for i in range(0, len(X), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_X = X[batch_indices]
                batch_Y = Y[batch_indices]
                loss = self.update(batch_X, batch_Y, lr)

                if i // batch_size % 10 == 0:
                    lr = utils.lr_decay(lr, epoch * len(X) + i, weight_decay=weight_decay, start_up=start_up,
                                        min_lr=min_lr)
                    losses.append(loss)
                if save:
                    if i // batch_size % 1000 == 0:
                        acc = self.test(X_test, Y_test)
                        accs.append(acc)
                        if acc > max_acc:
                            para = self.para()
                            config = self.config
                            max_acc = acc

                        print("Epoch:", epoch + 1, "\tLoss:", loss, "\tStep:", i, "\tAccuracy:", acc)

        acc = self.test(X_test, Y_test)
        accs.append(acc)
        if save:
            if i // batch_size % 1000 == 0:
                acc = self.test(X_test, Y_test)
                accs.append(acc)
                if acc > max_acc:
                    para = self.para()
                    config = self.config



        # 保存 loss和val_acc和最佳模型
        if save:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)

            with open(os.path.join(output_dir, 'loss.json'), 'w') as f:
                json.dump(losses, f)
            with open(os.path.join(output_dir, 'val_acc.json'), 'w') as f:
                json.dump(accs, f)
                path = os.path.join(output_dir, 'model.npy')
                np.save(path, {'para': para, 'config': config})


def load_model(model_path):
    para_config = np.load(model_path, allow_pickle=True).item()
    config = para_config['config']
    para = para_config['para']
    model = Model(hidden_layers=config['hidden_layers'], activation=config['activation'], lambda_=config['lambda_'])
    for i in range(len(para) * 2)[::2]:
        model.layers[i].weights = para[i // 2]['weights']
        model.layers[i].bias = para[i // 2]['bias']
    return model
