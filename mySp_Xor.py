import torch

import mySpikeProp as mySp


def Xor():
    # 网络结构定义
    ly1 = mySp.mySpikeProp(inCh=2, outCh=8)
    ly2 = mySp.mySpikeProp(inCh=8, outCh=2)

    '''print("初始：")
    print(ly1.e_dd)
    print(ly2.e_da)
    print("")'''

    # 数据准备
    net_in = torch.tensor([[0.01, 0.01], [0.01, 0.99], [0.99, 0.01], [0.99, 0.99]])  # 数据 (bs, 2)
    tar = torch.tensor([[0], [1], [1], [0]])  # 标签 (bs, 1)
    z0 = torch.exp(1.0 - net_in)  # 将输入编码为时间
    tar_2 = torch.ones(tar.size()[0], 2) * 0.01
    for i in range(z0.size()[0]):
        tar_2[i, tar[i]] = 0.99

    lr_Edd, lr_Eda = 1e-2, 1e-3  # 学习率
    result = []
    for epoch in range(200):
        # 前向传播过程
        z1 = ly1.forward(z0)  # (bs, 10)
        z2 = ly2.forward(z1)

        # 反向传播过程
        bs = z0.size()[0]
        lo = torch.softmax(z2, dim=1)
        tar_sp = torch.softmax(torch.exp(1.0 - tar_2), dim=1)  # 脉冲形式的标签

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

        # print("对异或的标签及结果：")
        # print("标签：" + str(torch.reshape(tar, [4])))
        result = torch.argmin(z2, dim=1)
        # print("输出：" + str(result))
        # print(" ")

    '''print("训练后：")
    print(ly1.e_dd)
    print(ly2.e_da)
    print("")'''

    if result.equal(torch.reshape(tar, [4])):
        return 1
    else:
        return 0


def main():
    correct = 0
    total_num = 1000
    for i in range(total_num):
        if i % int(total_num/10) == 0:
            print("当前次数"+str(i+1)+".")
        correct += Xor()
    print("正确率：" + str(int(correct)) + "/" + str(total_num), end="")
    print("(%.3f %%)" % (100. * correct / total_num))
    # Xor()


if __name__ == "__main__":
    main()
