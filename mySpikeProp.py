import torch


class mySpikeProp:
    def __init__(self, inCh, outCh, e_dd=None, e_da=None):
        self.inCh, self.outCh = inCh, outCh  # 输入、输出通道数
        self.th = 1.0
        if e_dd is None:  # 替代权值的树突延迟
            # self.e_dd = torch.rand(self.inCh, self.outCh) + torch.ones(self.inCh, self.outCh)  # 均匀分布初始化
            self.e_dd = torch.rand(self.inCh, self.outCh) * (8.0 / self.inCh)
        else:
            self.e_dd = e_dd
        if e_da is None:  # 输出的轴突延迟
            self.e_da = torch.ones(outCh)
        else:
            self.e_da = e_da
        self.cause_mask = torch.tensor(0)  # 引发集标注
        # Adam相关
        self.b1, self.b2, self.ep = 0.9, 0.9, 1e-8
        self.t, self.adam_m_Edd, self.adam_v_Edd = 0, torch.zeros_like(self.e_dd), torch.zeros_like(self.e_dd)
        self.adam_m_Eda, self.adam_v_Eda = torch.zeros_like(self.e_da), torch.zeros_like(self.e_da)

    def forward(self, lyIn, dv=torch.device("cpu")):
        bs, inCh, outCh = lyIn.size()[0], self.inCh, self.outCh  # 批大小、输入通道、输出通道

        inSt, inStInd = torch.sort(lyIn, dim=1)  # 发射时间从早到晚排序
        inStEx = torch.tile(torch.reshape(inSt, [bs, inCh, 1]), [1, 1, outCh])  # 输入扩展
        inStIndEx = torch.tile(torch.reshape(inStInd, [bs, inCh, 1]), [1, 1, outCh])  # 输入序号扩展
        EddStEx = torch.gather(torch.tile(torch.reshape(self.e_dd, [1, inCh, outCh]), [bs, 1, 1]),
                               dim=1, index=inStIndEx)  # 在维度1按发射顺序排列各突触并扩展
        Edd_inMul = EddStEx * inStEx  # 延迟作用在输入上

        CSiz = torch.cumsum(EddStEx, dim=1)  # 分母部分的逐项和
        Edd_inMulSum = torch.cumsum(Edd_inMul, dim=1)  # 分子部分的逐项和
        z_outAll = Edd_inMulSum / torch.clamp(CSiz - self.th, 1e-10, 1e10)  # 关键步，带正则化
        z_outAll = torch.where(CSiz < self.th, 1e5 * torch.ones_like(z_outAll), z_outAll)  # 特殊情况的处理（分母小于0）
        z_outAll = torch.where(z_outAll < inStEx, 1e5 * torch.ones_like(z_outAll), z_outAll)  # 特殊情况的处理（输出在输入前）
        z_out, z_outInd = torch.min(z_outAll, dim=1)  # 输出发射第一个脉冲的时间（同时确定了引发集）

        # 记录引发集
        _, inStIndR = torch.sort(inStInd, dim=1)  # 为恢复顺序
        z_outIndEx = torch.tile(torch.reshape(z_outInd, [bs, 1, outCh]), [1, inCh, 1])
        locEx = torch.tile(torch.reshape(torch.arange(inCh), [1, inCh, 1]), [bs, 1, outCh]).to(dv)
        self.cause_mask = torch.where(locEx <= z_outIndEx, 1, 0)  # 求到排序后的引发集
        self.cause_mask = torch.gather(self.cause_mask, dim=1, index=torch.tile(
            torch.reshape(inStIndR, [bs, self.inCh, 1]), [1, 1, self.outCh]))  # 恢复到原顺序的引发集（与突触对应）

        return z_out * self.e_da

    def backward(self, adj_Edd, adj_Eda, lr_Edd, lr_Eda):
        self.t += 1
        self.adam_m_Edd = self.b1 * self.adam_m_Edd + (1.0 - self.b1) * adj_Edd
        self.adam_v_Edd = self.b2 * self.adam_v_Edd + (1.0 - self.b2) * adj_Edd * adj_Edd
        M_Edd = self.adam_m_Edd / (1.0 - self.b1 ** self.t)
        V_Edd = self.adam_v_Edd / (1.0 - self.b2 ** self.t)
        self.e_dd -= lr_Edd * (M_Edd / (torch.sqrt(V_Edd) + self.ep))

        self.adam_m_Eda = self.b1 * self.adam_m_Eda + (1.0 - self.b1) * adj_Eda
        self.adam_v_Eda = self.b2 * self.adam_v_Eda + (1.0 - self.b2) * adj_Eda * adj_Eda
        M_Eda = self.adam_m_Eda / (1.0 - self.b1 ** self.t)
        V_Eda = self.adam_v_Eda / (1.0 - self.b2 ** self.t)
        self.e_da = self.e_da - lr_Eda * (M_Eda / (torch.sqrt(V_Eda) + self.ep))
