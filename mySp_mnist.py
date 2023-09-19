import math
import time
import torch

import Dataset as Ds
import mySpikeProp as mySp


def train(sn, bs):
    gpu_num = torch.cuda.device_count()
    if gpu_num == 0:
        print("程序用cpu训练。")
        device = torch.device("cpu")
    else:
        print("程序用gpu训练。")
        device = torch.device("cuda:0")

    # 网络结构的定义
    ly1 = mySp.mySpikeProp(inCh=784, outCh=800)
    ly2 = mySp.mySpikeProp(inCh=800, outCh=10)

    # 权重放到设备上
    ly1.e_dd, ly1.e_da = ly1.e_dd.to(device), ly1.e_da.to(device)
    ly2.e_dd, ly2.e_da = ly2.e_dd.to(device), ly2.e_da.to(device)
    ly1.cause_mask = ly1.cause_mask.to(device)
    ly2.cause_mask = ly2.cause_mask.to(device)
    ly1.adam_m_Edd, ly1.adam_m_Eda = ly1.adam_m_Edd.to(device), ly1.adam_m_Eda.to(device)
    ly2.adam_m_Edd, ly2.adam_m_Eda = ly2.adam_m_Edd.to(device), ly2.adam_m_Eda.to(device)
    ly1.adam_v_Edd, ly1.adam_v_Eda = ly1.adam_v_Edd.to(device), ly1.adam_v_Eda.to(device)
    ly2.adam_v_Edd, ly2.adam_v_Eda = ly2.adam_v_Edd.to(device), ly2.adam_v_Eda.to(device)

    # 数据准备
    X_train, y_train = [], []
    for idx, (data, target) in enumerate(Ds.fashion_mnist_train_loader):  # 一次读出全部数据
        X_train, y_train = data, target
    # X_train, y_train = Ds.mnist_train_border_detection()
    X_train = torch.where(X_train > 0.5, 0.01, 2.3)
    X_train, y_train = X_train.to(device), y_train.to(device)

    # 训练过程
    epoch_num = 40
    lr_start, lr_end = 1e-4, 1e-6
    lr_decay = (lr_end / lr_start) ** (1.0 / epoch_num)
    # lr_Eda = 1e-4

    bn = int(math.ceil(sn / bs))
    loss, total_loss = 0, []
    time_start = time.time()  # 记录训练开始时的时间
    for epoch in range(epoch_num):  # 6000
        lr_Edd = lr_start * lr_decay ** epoch
        for bi in range(bn):  # 20
            # 输入网络的数据定义
            if (bi + 1) * bs > sn:
                data, tar = X_train[bi * bs:sn], y_train[bi * bs:sn]
            else:
                data, tar = X_train[bi * bs:(bi + 1) * bs], y_train[bi * bs:(bi + 1) * bs]
            z0 = torch.exp(1.0 - data.view(-1, 28 * 28))  # 处理好了的数据 (bs,1,28,28) --> (bs,784)
            tar_10 = (torch.ones(tar.size()[0], 10)*0.99).to(device)  # 处理好了的标签
            for i in range(data.size()[0]):
                tar_10[i, tar[i]] = 0.01

            # 网络的前向输入过程
            z1 = ly1.forward(z0, dv=device)
            z2 = ly2.forward(z1, dv=device)

            # 反向传播过程
            bs = z0.size()[0]
            lo = torch.softmax(z2, dim=1)
            tar_sp = torch.softmax(torch.exp(tar_10), dim=1)

            # 后一层
            inCh, outCh = ly2.inCh, ly2.outCh
            e_dd2 = torch.tile(torch.reshape(ly2.e_dd, [1, inCh, outCh]), [bs, 1, 1])  # 附加
            e_da2Ex = torch.tile(torch.reshape(torch.exp(ly2.e_da), [1, 1, outCh]), [bs, inCh, 1])
            delta2 = lo - tar_sp
            delta2Ex = torch.tile(torch.reshape(delta2, [bs, 1, outCh]), [1, inCh, 1])  # 反传项
            z1Ex = torch.tile(torch.reshape(z1, [bs, inCh, 1]), [1, 1, outCh])  # z1
            z2Ex = torch.tile(torch.reshape(z2, [bs, 1, outCh]), [1, inCh, 1])  # z2
            T2 = (z1Ex - z2Ex) * ly2.cause_mask  # 当前项（指数时间差）
            adj_dd2 = torch.sum(delta2Ex * e_da2Ex * T2, dim=0)  # 关于树突延迟的项 delta2Ex * e_da2Ex * T2
            adj_da2 = torch.sum(delta2 * z2, dim=0)  # 关于轴突延迟的项 delta2 * z2

            # 前一层
            inCh, outCh = ly1.inCh, ly1.outCh
            e_da1Ex = torch.tile(torch.reshape(torch.exp(ly1.e_da), [1, 1, outCh]), [bs, inCh, 1])
            delta1 = torch.sum(delta2Ex * ly2.cause_mask * e_dd2, dim=2)  # delta2Ex * ly2.cause_mask * e_dd2
            delta1Ex = torch.tile(torch.reshape(delta1, [bs, 1, outCh]), [1, inCh, 1])  # 反传项
            z0Ex = torch.tile(torch.reshape(z0, [bs, inCh, 1]), [1, 1, outCh])
            z1Ex = torch.tile(torch.reshape(z1, [bs, 1, outCh]), [1, inCh, 1])
            T1 = (z0Ex - z1Ex) * ly1.cause_mask  # 当前项（指数时间差）
            adj_dd1 = torch.sum(delta1Ex * e_da1Ex * T1, dim=0)  # 关于树突延迟的项 delta1Ex * e_da1Ex * T1
            adj_da1 = torch.sum(delta1 * z1, dim=0)  # 关于轴突延迟的项 delta1 * z1

            # 更新过程
            ly2.backward(adj_dd2, adj_da2, lr_Edd, lr_Edd)
            ly1.backward(adj_dd1, adj_da1, lr_Edd, lr_Edd)

            CE = -1.0 * torch.sum(torch.log(torch.clamp(lo, 1e-5, 1.0)) * tar_sp) / data.size()[0]
            CE_min = -1.0 * torch.sum(torch.log(torch.clamp(tar_sp, 1e-5, 1.0)) * tar_sp) / data.size()[0]
            loss = abs(CE-CE_min)
            if bi % 10 == 0:
                print("训练轮数：" + str(epoch + 1), end="\t")
                print("进度：[" + str(bi * bs) + "/" + str(sn), end="")
                print("(%.0f %%)]" % (100.0 * bi * bs / sn), end="\t")
                print("误差：" + str(loss))
                total_loss.append(loss)
        pass
        time_epoch_end = time.time()
        print("至轮结束耗时：%.3f s" % (time_epoch_end - time_start))
        torch.save(ly1.e_dd, "./parameters_record/mySp_mnist_edd1")
        torch.save(ly1.e_da, "./parameters_record/mySp_mnist_eda1")
        torch.save(ly2.e_dd, "./parameters_record/mySp_mnist_edd2")
        torch.save(ly2.e_da, "./parameters_record/mySp_mnist_eda2")
        print("测试集上误差：")
        test(1000, 100)
    pass
    time_end = time.time()  # 记录训练结束时的时间
    print("训练耗时：%.3f s" % (time_end - time_start))
    print(torch.tensor(total_loss).size())
    print(torch.tensor(total_loss))


def test(sn, bs):
    gpu_num = torch.cuda.device_count()
    if gpu_num == 0:
        print("程序用cpu测试。")
        device = torch.device("cpu")
    else:
        print("程序用gpu测试。")
        device = torch.device("cuda:0")

    e_dd1 = torch.load("./parameters_record/mySp_mnist_edd1").to(device)
    e_da1 = torch.load("./parameters_record/mySp_mnist_eda1").to(device)
    e_dd2 = torch.load("./parameters_record/mySp_mnist_edd2").to(device)
    e_da2 = torch.load("./parameters_record/mySp_mnist_eda2").to(device)

    ly1 = mySp.mySpikeProp(784, 800, e_dd=e_dd1, e_da=e_da1)
    ly2 = mySp.mySpikeProp(800, 10, e_dd=e_dd2, e_da=e_da2)

    # 数据准备
    X_test, y_test = [], []
    for idx, (data, target) in enumerate(Ds.fashion_mnist_test_loader):
        X_test, y_test = data, target
    # X_test, y_test = Ds.mnist_test_border_detection()
    X_test = torch.where(X_test > 0.5, 0.01, 2.3)
    X_test, y_test = X_test.to(device), y_test.to(device)

    # 测试过程
    correct = 0
    bn = int(math.ceil(sn / bs))
    for bi in range(bn):
        if (bi + 1) * bs > sn:
            data, tar = X_test[bi * bs:sn], y_test[bi * bs:sn]
        else:
            data, tar = X_test[bi * bs:(bi + 1) * bs], y_test[bi * bs:(bi + 1) * bs]
        z0 = torch.exp(1.0 - data.view(-1, 28 * 28))

        # 网络的前向输入过程
        z1 = ly1.forward(z0, dv=device)
        z2 = ly2.forward(z1, dv=device)

        lo = torch.softmax(z2, dim=1)
        prediction = torch.argmin(lo, dim=1)
        correct += prediction.eq(tar.data).sum()
    pass
    print("正确率：" + str(int(correct)) + "/" + str(sn), end="")
    print("(%.3f %%)" % (100. * correct / sn))


def main():
    train(60000, 128)
    test(10000, 100)


if __name__ == "__main__":
    main()
