import load_data
import model
import matplotlib.pyplot as plt
import json

model_path = './output/final_model_2/model.npy'
loss_path = './output/final_model_2/loss.json'
accuracy_path = './output/final_model_2/val_acc.json'

X_test, y_test = load_data.load_mnist('./data', kind='t10k')
final_model = model.load_model(model_path)
with open(loss_path, 'r') as f:
    loss = json.load(f)
with open(accuracy_path, 'r') as f:
    accuracy = json.load(f)

img = plt.figure(1)
x = range(len(loss)*10)[::10]
plt.plot(x, loss)
plt.xlabel('steps')
plt.ylabel('train loss')
plt.savefig('./output/figure/train_loss_2.png')

img = plt.figure(2)
x = range(len(accuracy)*1000)[::1000]
plt.plot(x, accuracy)
plt.xlabel('steps')
plt.ylabel('valid accuracy')
plt.savefig('./output/figure/test_accuracy_2.png')

print('test accuracy:', final_model.test(X_test, y_test))
