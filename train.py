import load_data
import model
import matplotlib.pyplot as plt
import os

X_train, y_train = load_data.load_mnist('./data', kind='train')
X_valid, y_valid = X_train[:10000], y_train[:10000]
X_train, y_train = X_train[10000:], y_train[10000:]
X_test, y_test = load_data.load_mnist('./data', kind='t10k')

os.makedirs('./output/figure5', exist_ok=True)


def best_lr(lrs=[0.5, 0.2, 0.1, 0.05, 0.01, 0.001], epoch=15):
    accs = []
    for lr in lrs:
        model_lr = model.Model(lambda_=0, hidden_layers=(512, 256))
        model_lr.train(X_train, y_train, X_valid, y_valid, batch_size=32, weight_decay=1 - 1e-4, epochs=epoch, lr=lr,
                       start_up=200, min_lr=1e-4)
        acc = model_lr.test(X_valid, y_valid)
        accs.append(acc)

    img = plt.figure(1)
    x = range(len(lrs))
    plt.plot(x, accs, marker='o')
    plt.xlabel('learning rate')
    plt.ylabel('test_accuracy')
    plt.xticks(x, lrs)
    plt.savefig('./output/figure5/lr_accuracy.png')
    return lrs[accs.index(max(accs))]


def best_lambda(lr=0.05, lambdas=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 0], epoch=15):
    accs = []
    for lambda_ in lambdas:
        model_lambda = model.Model(lambda_=lambda_, hidden_layers=(512, 256))
        model_lambda.train(X_train, y_train, X_valid, y_valid, batch_size=32, weight_decay=1 - 1e-4, epochs=epoch,
                           lr=lr,
                           start_up=200, min_lr=1e-4)
        acc = model_lambda.test(X_valid, y_valid)
        accs.append(acc)

    img = plt.figure(2)
    x = range(len(lambdas))
    plt.plot(x, accs, marker='o')
    plt.xlabel('lambda')
    plt.ylabel('test_accuracy')
    plt.xticks(x, lambdas)
    plt.savefig('./output/figure5/lambda_accuracy.png')

    return lambdas[accs.index(max(accs))]


def different_hidden_layer(lr=0.05, lambda_=0, hidden_layers=[512, 256, 128, 64, 32, 16], i=1, other_layer=512,
                           epoch=15):
    accs = []
    for hidden_layer in hidden_layers:
        h = [other_layer, other_layer]
        h[i] = hidden_layer
        model_hidden = model.Model(lambda_=lambda_, hidden_layers=tuple(h))
        model_hidden.train(X_train, y_train, X_valid, y_valid, batch_size=32, weight_decay=1 - 1e-4, epochs=epoch,
                           lr=lr,
                           start_up=200, min_lr=1e-4)
        acc = model_hidden.test(X_valid, y_valid)
        accs.append(acc)

    img = plt.figure(2 + i + 1)
    x = range(len(hidden_layers))
    plt.plot(x, accs, marker='o')
    plt.xlabel('hidden dim')
    plt.ylabel('test_accuracy')
    plt.xticks(x, hidden_layers)
    plt.savefig('./output/figure5/hidden' + str(i) + '_accuracy' + '.png')

    return hidden_layers[accs.index(max(accs))]


def main():
    epoch = 20
    lr = best_lr()
    lambda_ = best_lambda(lr=lr, epoch=epoch)
    h2 = different_hidden_layer(lr=lr, lambda_=lambda_, other_layer=512, epoch=epoch)
    h1 = different_hidden_layer(lr=lr, lambda_=lambda_, hidden_layers=[784, 512, 256, 128, 64], i=0, other_layer=h2,
                                epoch=epoch)
    best_model = model.Model(lambda_=lambda_, hidden_layers=(h1, h2))
    best_model.train(X_train, y_train, X_valid, y_valid, batch_size=32, weight_decay=1 - 1e-4, epochs=epoch, lr=lr,
                     start_up=200, min_lr=1e-5, save=True, output_dir='./output/final_model_4')


if __name__ == '__main__':
    main()
