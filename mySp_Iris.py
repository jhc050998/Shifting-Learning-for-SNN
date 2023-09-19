import math
import time
import torch

import Dataset as Ds
import mySpikeProp as mySp


def Iris():
    # 数据准备
    X, y = Ds.iris_loader()

    time_start = time.time()  # 记录训练开始时的时间
    total_correct = 0
    for f in range(5):  # 5折交叉验证
        print("第" + str(f+1) + "次。")
        # 网络结构定义
        ly1 = mySp.mySpikeProp(inCh=4, outCh=10)
        ly2 = mySp.mySpikeProp(inCh=10, outCh=3)

        # 取出数据
        X_train, y_train = torch.cat((X[0:f*30], X[f*30+30:300]), dim=0), torch.cat((y[0:f*30], y[f*30+30:300]), dim=0)
        X_test, y_test = X[f*30:f*30+30], y[f*30:f*30+30]

        # 训练过程
        epoch_num = 400
        lr_start, lr_end = 1e-2, 1e-4
        lr_decay = (lr_end / lr_start) ** (1.0 / epoch_num)
        lr_Eda = 1e-4  # 学习率

        sn, bs = 120, 30  # 训练数据数、批大小
        bn = int(math.ceil(sn / bs))
        loss, total_loss = 0, []
        for epoch in range(epoch_num):
            lr_Edd = lr_start * lr_decay ** epoch
            for bi in range(bn):
                # 输入网络的数据定义
                if (bi + 1) * bs > sn:
                    data, tar_3 = X_train[bi * bs:sn], y_train[bi * bs:sn]
                else:
                    data, tar_3 = X_train[bi * bs:(bi + 1) * bs], y_train[bi * bs:(bi + 1) * bs]
                z0 = torch.exp(1.0 - data)  # (30,4)

                # 网络的前向输入过程
                z1 = ly1.forward(z0)
                z2 = ly2.forward(z1)

                # 反向传播过程
                bs = z0.size()[0]
                lo = torch.softmax(z2, dim=1)
                tar_sp = torch.softmax(torch.exp(1.0 - tar_3), dim=1)

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
                ly2.backward(adj_dd2, adj_da2, lr_Edd, lr_Eda)
                ly1.backward(adj_dd1, adj_da1, lr_Edd, lr_Eda)

                CE = -1.0 * torch.sum(torch.log(torch.clamp(lo, 1e-5, 1.0)) * tar_sp) / data.size()[0]
                CE_min = -1.0 * torch.sum(torch.log(torch.clamp(tar_sp, 1e-5, 1.0)) * tar_sp) / data.size()[0]
                loss = abs(CE - CE_min)
            pass
            if epoch % 1 == 0:
                '''print("训练轮数：" + str(epoch + 1), end="\t")
                print("进度：[" + str(epoch) + "/" + str(epoch_num), end="")
                print("(%.0f %%)]" % (100.0 * epoch / epoch_num), end="\t")
                print(" ")
                print("误差：" + str(loss))'''
                total_loss.append(loss)
        pass

        # print(ly1.e_dd)
        # print(ly2.e_da)
        # print("")

        total_loss = torch.tensor(total_loss)
        # print(total_loss)

        # 测试过程
        correct = 0
        sn, bs = 30, 30  # 训练数据数、批大小
        bn = int(math.ceil(sn / bs))
        for bi in range(bn):
            if (bi + 1) * bs > sn:
                data, tar_3 = X_test[bi * bs:sn], y_test[bi * bs:sn]
            else:
                data, tar_3 = X_test[bi * bs:(bi + 1) * bs], y_test[bi * bs:(bi + 1) * bs]
            z0 = torch.exp(1.0 - data)  # (30,4)
            tar = torch.argmax(tar_3, dim=1)

            # 网络的前向输入过程
            z1 = ly1.forward(z0)
            z2 = ly2.forward(z1)

            lo = z2
            prediction = torch.argmin(lo, dim=1)
            correct += prediction.eq(tar.data).sum()
        pass
        total_correct += int(correct)
        print("正确率：" + str(int(correct)) + "/" + str(sn), end="")
        print("(%.3f %%)" % (100. * correct / sn))
        print(" ")
    pass
    time_end = time.time()  # 记录训练结束时的时间

    print(" ")
    print("总正确率：" + str(total_correct) + "/" + str(150), end="")
    print("(%.3f %%)" % (100. * total_correct / 150))
    print('训练耗时：%.3f s' % (time_end - time_start))


def main():
    Iris()


if __name__ == "__main__":
    main()
