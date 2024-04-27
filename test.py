import load_data
import model

model_path = './output/final_model/'

model_path = model_path + 'model.npy'


X_test, y_test = load_data.load_mnist('./data', kind='t10k')
final_model = model.load_model(model_path)

X_train, y_train = load_data.load_mnist('./data', kind='train')
X_valid, y_valid = X_train[:10000], y_train[:10000]
X_train, y_train = X_train[10000:], y_train[10000:]

print('test accuracy:', final_model.test(X_test, y_test))
print('train accuracy:', final_model.test(X_train, y_train))
print('valid accuracy:', final_model.test(X_valid, y_valid))
