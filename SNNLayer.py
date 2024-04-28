import torch


class SNNLayer:
    def __init__(self, inCh, outCh, e_ts=None, e_tp=None):
        self.inCh, self.outCh = inCh, outCh  # number of input, output channels
        self.th = 1.0
        if e_ts is None:  # exponential shifting timings
            self.e_ts = torch.rand(self.inCh, self.outCh) * (8.0 / self.inCh)  # initialized by uniform distribution
        else:
            self.e_ts = e_ts
        if e_tp is None:  # exponential passing time between neurons added to the output
            self.e_tp = torch.ones(outCh)
        else:
            self.e_tp = e_tp
        self.cause_mask = torch.tensor(0)  # casual set record
        # Adam related parameters
        self.b1, self.b2, self.ep = 0.9, 0.9, 1e-8
        self.t, self.adam_m_Ets, self.adam_v_Ets = 0, torch.zeros_like(self.e_ts), torch.zeros_like(self.e_ts)
        self.adam_m_Etp, self.adam_v_Etp = torch.zeros_like(self.e_tp), torch.zeros_like(self.e_tp)

    def forward(self, bs, z_in, dv=torch.device("cpu")):  # IF
        bs, inCh, outCh = bs, self.inCh, self.outCh  # batch size, channels

        z_inSt, z_inStInd = torch.sort(z_in, dim=1)  # sort the firing times from early to late
        z_inStEx = torch.tile(torch.reshape(z_inSt, [bs, inCh, 1]), [1, 1, outCh])  # expand z_in for GPU computing
        z_inStIndEx = torch.tile(torch.reshape(z_inStInd, [bs, inCh, 1]), [1, 1, outCh])  # expand the indexes
        EtsStEx = torch.gather(torch.tile(torch.reshape(self.e_ts, [1, inCh, outCh]), [bs, 1, 1]),
                               dim=1, index=z_inStIndEx)  # arrange and expand e_dd to match the sorted, expanded z_in
        EtsZinMul = EtsStEx * z_inStEx  # shifting timings added to input times

        EtsSum = torch.cumsum(EtsStEx, dim=1)  # sum of e_ts in C for denominator
        EtsZinMulSum = torch.cumsum(EtsZinMul, dim=1)  # cumulative sum of e_ts*z_in for numerator
        z_outAll = EtsZinMulSum / torch.clamp(EtsSum - self.th, 1e-10, 1e10)  # all assumed outputs
        z_outAll = torch.where(EtsSum < self.th, 1e5 * torch.ones_like(z_outAll), z_outAll)  # exclude denominator = 0
        z_outAll = torch.where(z_outAll < z_inStEx, 1e5 * torch.ones_like(z_outAll), z_outAll)  # exclude in after out
        z_out, z_outInd = torch.min(z_outAll, dim=1)  # the first assumed output is the real output

        # Complete casual set record
        _, inStIndR = torch.sort(z_inStInd, dim=1)  # for reverse the order
        z_outIndEx = torch.tile(torch.reshape(z_outInd, [bs, 1, outCh]), [1, inCh, 1])
        locEx = torch.tile(torch.reshape(torch.arange(inCh), [1, inCh, 1]), [bs, 1, outCh]).to(dv)
        self.cause_mask = torch.where(locEx <= z_outIndEx, 1, 0)  # record of casual set after sorting
        self.cause_mask = torch.gather(self.cause_mask, dim=1, index=torch.tile(
            torch.reshape(inStIndR, [bs, self.inCh, 1]), [1, 1, self.outCh]))  # record of casual set before sorting

        return z_out * self.e_tp

    def pass_delta(self, bs, delta):  # Supervised Term
        bs, inCh, outCh = bs, self.inCh, self.outCh  # batch size, channels
        e_ts2 = torch.tile(torch.reshape(self.e_ts, [1, inCh, outCh]), [bs, 1, 1])
        delta2Ex = torch.tile(torch.reshape(delta, [bs, 1, outCh]), [1, inCh, 1])
        delta_out = torch.sum(delta2Ex * self.cause_mask * e_ts2, dim=2)  # delta2Ex * self.cause_mask * e_ts2
        return delta_out

    def backward(self, bs, delta, z_in, z_out, lr_Ets, lr_Etp):  # Shifting Learning
        bs, inCh, outCh = bs, self.inCh, self.outCh  # batch size, channels
        e_tpEx = torch.tile(torch.reshape(torch.exp(self.e_tp), [1, 1, outCh]), [bs, inCh, 1])
        deltaEx = torch.tile(torch.reshape(delta, [bs, 1, outCh]), [1, inCh, 1])  # Broadcast Term
        z_inEx = torch.tile(torch.reshape(z_in, [bs, inCh, 1]), [1, 1, outCh])  # z_in
        z_outEx = torch.tile(torch.reshape(z_out, [bs, 1, outCh]), [1, inCh, 1])  # z_out
        T = (z_inEx - z_outEx) * self.cause_mask  # Local Term (exponential time difference)
        adj_Ets = torch.sum(deltaEx * e_tpEx * T, dim=0)
        adj_Etp = torch.sum(delta * z_out, dim=0)

        # 不用自适应
        # self.e_ts -= lr_Ets * adj_Ets
        # self.e_tp -= lr_Etp * adj_Etp

        # Adam
        self.t += 1
        self.adam_m_Ets = self.b1 * self.adam_m_Ets + (1.0 - self.b1) * adj_Ets
        self.adam_v_Ets = self.b2 * self.adam_v_Ets + (1.0 - self.b2) * adj_Ets * adj_Ets
        M_Ets = self.adam_m_Ets / (1.0 - self.b1 ** self.t)
        V_Ets = self.adam_v_Ets / (1.0 - self.b2 ** self.t)
        self.e_ts -= lr_Ets * (M_Ets / (torch.sqrt(V_Ets) + self.ep))

        self.adam_m_Etp = self.b1 * self.adam_m_Etp + (1.0 - self.b1) * adj_Etp
        self.adam_v_Etp = self.b2 * self.adam_v_Etp + (1.0 - self.b2) * adj_Etp * adj_Etp
        M_Etp = self.adam_m_Etp / (1.0 - self.b1 ** self.t)
        V_Etp = self.adam_v_Etp / (1.0 - self.b2 ** self.t)
        self.e_tp = self.e_tp - lr_Etp * (M_Etp / (torch.sqrt(V_Etp) + self.ep))
