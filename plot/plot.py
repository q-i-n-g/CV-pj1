import load_data
import model
import matplotlib.pyplot as plt
import json
import os

model_path = './output/final_model/'
loss_path = model_path + 'loss.json'
val_loss_path = model_path + 'val_loss.json'
accuracy_path = model_path + 'val_acc.json'
model_path = model_path + 'model.npy'
save_path = './output/figure5/'

os.makedirs(save_path[:-1], exist_ok=True)

X_test, y_test = load_data.load_mnist('./data', kind='t10k')
final_model = model.load_model(model_path)
with open(loss_path, 'r') as f:
    loss = json.load(f)
with open(val_loss_path, 'r') as f:
    val_loss = json.load(f)
with open(accuracy_path, 'r') as f:
    accuracy = json.load(f)

img = plt.figure(1)
x = range(len(loss)*10)[::10]
plt.plot(x, loss)
plt.xlabel('steps')
plt.ylabel('train loss')
plt.savefig(save_path + 'train_loss.png')

img = plt.figure(0)
x = range(len(val_loss)*1000)[::1000]
plt.plot(x, val_loss)
plt.xlabel('steps')
plt.ylabel('val loss')
plt.savefig(save_path + 'val_loss.png')

img = plt.figure(2)
x = range(len(accuracy)*1000)[::1000]
plt.plot(x, accuracy)
plt.xlabel('steps')
plt.ylabel('valid accuracy')
plt.savefig(save_path + 'test_accuracy.png')



for i in range(6)[::2]:
    img = plt.figure(3 + i)
    weight = final_model.layers[i].weights
    plt.axis('off')
    vmin = weight.min()
    vmax = weight.max()
    plt.imshow(weight, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.savefig(save_path + 'weight' + str(i//2 + 1) + '.png')

    bias = final_model.layers[i].bias
    img = plt.figure(3 + i+1)
    vmin = bias.min()
    vmax = bias.max()
    plt.imshow(bias, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.savefig(save_path + 'bias' + str(i//2 + 1) + '.png')


fig, axes = plt.subplots(4, 4)
vmin, vmax = final_model.para()[0]['weights'].min(), final_model.para()[0]['weights'].max()
for coef, ax in zip(final_model.para()[0]['weights'].T, axes.ravel()):
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
               vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.savefig(save_path + 'w1.png')


