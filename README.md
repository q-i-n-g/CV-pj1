# CV-pj1
基于numpy手工搭建三层神经网络分类器, 在数据集[Fashion-MNIST]上实现图像分类.
## 数据准备
到https://github.com/zalandoresearch/fashion-mnist 下载数据集，并在根目录下创立data文件夹，将数据集下载到data中。
## 测试
下载模型权重，在根目录下新建output/final_model文件夹，将.pth文件放入其中，运行test.py文件即可完成对final model的测试。

修改路径后可以运行test.py完成任何模型测试。
## 训练
### quick start
运行quick_train.py，在默认超参下训练，也可以手动调整训练和模型参数。
### 参数查找
运行train.py可以自动实现参数查找，默认为训练20个epoch。
