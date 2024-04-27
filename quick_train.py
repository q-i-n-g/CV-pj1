import load_data
import model
import matplotlib.pyplot as plt
import os

X_train, y_train = load_data.load_mnist('./data', kind='train')
X_valid, y_valid = X_train[:10000], y_train[:10000]
X_train, y_train = X_train[10000:], y_train[10000:]
X_test, y_test = load_data.load_mnist('./data', kind='t10k')

output_dir = './output/final_model'
os.makedirs(output_dir, exist_ok=True)
model = model.Model()
model.train(X_train, y_train, X_valid, y_valid, batch_size=32, weight_decay=1 - 1e-4, epochs=50, lr=0.1,
                     start_up=200, min_lr=1e-5, save=True, output_dir=output_dir)
