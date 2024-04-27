import matplotlib.pyplot as plt
import load_data

X_train, y_train = load_data.load_mnist('./data', kind='train')

for i, img in enumerate(X_train[:36]):
    img = img.reshape(28, 28)
    plt.subplot(6, 6, i+1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])

plt.savefig('./output/data/examples.png')
plt.show()
