import numpy
import pandas

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from skimage import feature


# iris 鸢尾花数据集
def iris_loader():
    row_data = numpy.mat(pandas.read_csv("D:/Dataset/iris/iris.data", header=None, sep=','))  # 读取数据
    row_data1, row_data2, row_data3 = row_data[0:50, :], row_data[50:100, :], row_data[100:150, :]
    row_data1[:, 4], row_data2[:, 4], row_data3[:, 4] = 0, 1, 2  # 山鸢尾记为0, 变色鸢尾记为1, 维吉尼亚鸢尾记为2
    row_data = numpy.row_stack((row_data1, row_data2, row_data3))

    tensor_data = torch.from_numpy(row_data.astype(float))  # 打乱顺序
    rand_index = torch.tile(torch.reshape(torch.randperm(150), [150, 1]), [1, 5])
    tensor_data = torch.gather(tensor_data, dim=0, index=rand_index)

    tensor_tars = torch.ones((150, 3)) * 0.01  # 标签one-hot编码
    for i in range(150):
        tensor_tars[i, int(tensor_data[i, 4])] = 0.99

    tensor_inputs = tensor_data[:, 0:4]  # 输入正则化
    data_max, data_min = torch.max(tensor_inputs), torch.min(tensor_inputs)
    # data_max, _ = torch.max(tensor_inputs, dim=0)  # 返回值和序号，取值
    # data_min, _ = torch.min(tensor_inputs, dim=0)
    tensor_inputs = (tensor_inputs - data_min)/(data_max - data_min) + 0.01
    return tensor_inputs, tensor_tars


# wine 红酒数据集
def wine_loader(path, num):
    data_file = open(path, 'r')
    data_list = data_file.readlines()
    data_file.close()

    tensor_data = torch.zeros((num, 12))  # 数据作成torch张量
    row_data = numpy.array(data_list)[1:]  # (1599,)
    for i in range(num):
        data_item = row_data[i].split(";")  # (12,)
        for j in range(11):
            tensor_data[i, j] = float(data_item[j])
        tensor_data[i, 11] = float(data_item[11][0])

    rand_index = torch.tile(torch.reshape(torch.randperm(num), [num, 1]), [1, 12])  # 打乱顺序
    tensor_data = torch.gather(tensor_data, dim=0, index=rand_index)

    tensor_tars = torch.ones((num, 10)) * 0.01  # 标签one-hot编码
    for i in range(num):
        tensor_tars[i, int(tensor_data[i, 11])] = 0.99

    tensor_inputs = tensor_data[:, 0:11]  # 输入正则化
    data_max, _ = torch.max(tensor_inputs, dim=0)  # 返回值和序号，取值
    data_min, _ = torch.min(tensor_inputs, dim=0)
    tensor_inputs = (tensor_inputs - data_min) / (data_max - data_min) + 0.01

    return tensor_inputs, tensor_tars


# wdBC 威斯康星癌细胞数据集
def wdBC_loader():
    row_data = numpy.mat(pandas.read_csv("D:/Dataset/wdBC/wdBC.data", header=None, sep=','))  # 读取数据
    row_inputs, row_tars = row_data[:, 2:32], row_data[:, 1]

    tensor_inputs, tensor_tars = torch.from_numpy(row_inputs.astype(float)), torch.ones((569, 2))*0.01
    for i in range(569):  # 完成标签的one-hot编码
        if row_tars[i] == 'M':
            tensor_tars[i, 1] = 0.99
        elif row_tars[i] == 'B':
            tensor_tars[i, 0] = 0.99

    rand_index = torch.randperm(569)  # 重排顺序
    tensor_tars = torch.gather(tensor_tars, dim=0, index=torch.tile(torch.reshape(rand_index, [569, 1]), [1, 2]))
    tensor_inputs = torch.gather(tensor_inputs, dim=0, index=torch.tile(torch.reshape(rand_index, [569, 1]), [1, 30]))

    data_max, _ = torch.max(tensor_inputs, dim=0)  # 返回值和序号，取值
    data_min, _ = torch.min(tensor_inputs, dim=0)
    tensor_inputs = (tensor_inputs - data_min) / (data_max - data_min) + 0.01
    tot_max, tot_min = torch.max(tensor_inputs), torch.min(tensor_inputs)
    tensor_inputs = (tensor_inputs - tot_min) / (tot_max - tot_min) + 0.01
    # mean, var = torch.mean(tensor_inputs, dim=0), torch.var(tensor_inputs, dim=0)
    # tensor_inputs = (tensor_inputs - mean)/var
    return tensor_inputs, tensor_tars


# MNIST
mnist_train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("D:/Dataset/mnist", train=True, download=True,  # MNIST/raw中
                   transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=60000, shuffle=True  # 每个batch load的数据，是不是打乱每个batch的数据
)
mnist_test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("D:/Dataset/mnist", train=False, download=True,
                   transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=10000, shuffle=False  # 每个batch load的数据，是不是打乱每个batch的数据
)

# FashionMNIST
fashion_mnist_train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST("D:/Dataset/mnist", train=True, download=True,  # FashionMNIST/raw中
                          transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=60000, shuffle=True
)
fashion_mnist_test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST("D:/Dataset/mnist", train=False, download=True,
                          transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=10000, shuffle=False
)


def mnist_train_border_detection():
    X_train, y_train = [], []
    for idx, (data, target) in enumerate(mnist_train_loader):  # 一次读出全部数据
        X_train, y_train = data, target
    for item in X_train:
        item[0] = torch.tensor(feature.canny(numpy.array(item[0]), sigma=0.5))
    return X_train, y_train


def mnist_test_border_detection():
    X_test, y_test = [], []
    for idx, (data, target) in enumerate(mnist_test_loader):
        X_test, y_test = data, target
    for item in X_test:
        item[0] = torch.tensor(feature.canny(numpy.array(item[0]), sigma=0.5))
    return X_test, y_test


# CiFar10
CIFAR10_train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10("D:/Dataset/CiFar", train=True, download=True,
                     transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=50000, shuffle=True
)
CIFAR10_test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10("D:/Dataset/CiFar", train=False, download=True,
                     transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=10000, shuffle=False
)


# 展示MNIST & FashionMNIST数据集中的图像
def MNIST_show(data_set="mnist", show_part="train"):
    train_data_loader = mnist_train_loader
    if data_set == "fashion_mnist":
        train_data_loader = fashion_mnist_train_loader
    test_data_loader = mnist_test_loader
    if data_set == "fashion_mnist":
        test_data_loader = fashion_mnist_test_loader

    X_train, y_train = [], []
    for idx, (data, target) in enumerate(train_data_loader):  # 一次读出全部数据
        X_train, y_train = data, target
    print("训练集数据形式：" + str(X_train.shape))  # (60000,1,28,28)
    print("训练集标签形式：" + str(y_train.shape))  # (60000)
    print("")

    X_test, y_test = [], []
    for idx, (data, target) in enumerate(test_data_loader):
        X_test, y_test = data, target
    print("测试集数据形式：" + str(X_test.shape))  # (10000,1,28,28)
    print("测试集标签形式：" + str(y_test.shape))  # (10000)
    print("")

    if show_part == "train":
        for i in range(X_train.size()[0]):   # 展示训练集
            data, target = X_train[i], y_train[i]
            print("图" + str(i+1) + "标签：" + str(target))
            img = data.permute(1, 2, 0)
            x_img = data[0]
            x_img = torch.where(x_img > 0.5, 0.01, 1.79)
            # x_img = numpy.array(data[0])
            # x_img = feature.canny(x_img, sigma=0.5)  # 边缘提取
            print(x_img)
            plt.imshow(x_img)
            plt.show()
    else:
        for i in range(X_test.size()[0]):  # 展示测试集
            data, target = X_test[i], y_test[i]
            print("图" + str(i+1) + "标签" + str(target))
            img = data.permute(1, 2, 0)
            plt.imshow(img)
            plt.show()


# 展示CiFar10数据集中的图像
def CiFar_show(show_part="train"):
    train_data_loader = CIFAR10_train_loader
    test_data_loader = CIFAR10_test_loader

    X_train, y_train = [], []
    for idx, (data, target) in enumerate(train_data_loader):  # 一次读出全部数据
        X_train, y_train = data, target
    print("训练集数据形式：" + str(X_train.shape))  # (50000,3,32,32)
    print("训练集标签形式：" + str(y_train.shape))  # (50000)
    print("")

    X_test, y_test = [], []
    for idx, (data, target) in enumerate(test_data_loader):
        X_test, y_test = data, target
    print("测试集数据形式：" + str(X_test.shape))  # (10000,3,32,32)
    print("测试集标签形式：" + str(y_test.shape))  # (10000)
    print("")

    if show_part == "train":
        for i in range(X_train.size()[0]):  # 展示训练集
            data, target = X_train[i], y_train[i]
            print("图" + str(i + 1) + "标签：" + str(target))
            img = data.permute(1, 2, 0)
            plt.imshow(img)
            plt.show()
    else:
        for i in range(X_test.size()[0]):  # 展示测试集
            data, target = X_test[i], y_test[i]
            print("图" + str(i + 1) + "标签" + str(target))
            img = data.permute(1, 2, 0)
            plt.imshow(img)
            plt.show()


def main():
    MNIST_show(data_set="mnist", show_part="train")
    # CiFar_show(show_part="test")

    # mnist_train_border_detection()

    # data = iris_loader()
    # iris_norm(data, 7)
    # wine_loader("D:/Dataset/wine/wine-quality-red.csv", 1599)
    # wine_loader("D:/Dataset/wine/wine-quality-white.csv", 4898)
    # wine_white_loader()
    # wdBC_loader()


if __name__ == "__main__":
    main()
