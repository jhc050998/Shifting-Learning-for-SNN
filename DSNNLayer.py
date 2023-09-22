import torch


class DSNNLayer:
    def __init__(self, inCh, outCh, e_dd=None, e_da=None):
        self.inCh, self.outCh = inCh, outCh  # number of input, output channels
        self.th = 1.0
        if e_dd is None:  # exponential dendrite delays replacing weights
            self.e_dd = torch.rand(self.inCh, self.outCh) * (8.0 / self.inCh)  # initialized by uniform distribution
        else:
            self.e_dd = e_dd
        if e_da is None:  # exponential axon delays added to the output
            self.e_da = torch.ones(outCh)
        else:
            self.e_da = e_da
        self.cause_mask = torch.tensor(0)  # casual set record
        # Adam related parameters
        self.b1, self.b2, self.ep = 0.9, 0.9, 1e-8
        self.t, self.adam_m_Edd, self.adam_v_Edd = 0, torch.zeros_like(self.e_dd), torch.zeros_like(self.e_dd)
        self.adam_m_Eda, self.adam_v_Eda = torch.zeros_like(self.e_da), torch.zeros_like(self.e_da)

    def forward(self, bs, z_in, dv=torch.device("cpu")):  # DSNN-IF
        bs, inCh, outCh = bs, self.inCh, self.outCh  # batch size, channels

        z_inSt, z_inStInd = torch.sort(z_in, dim=1)  # sort the firing times from early to late
        z_inStEx = torch.tile(torch.reshape(z_inSt, [bs, inCh, 1]), [1, 1, outCh])  # expand z_in for GPU computing
        z_inStIndEx = torch.tile(torch.reshape(z_inStInd, [bs, inCh, 1]), [1, 1, outCh])  # expand the indexes
        EddStEx = torch.gather(torch.tile(torch.reshape(self.e_dd, [1, inCh, outCh]), [bs, 1, 1]),
                               dim=1, index=z_inStIndEx)  # arrange and expand e_dd to match the sorted, expanded z_in
        EddZinMul = EddStEx * z_inStEx  # dendrite delay added to input times

        EddSum = torch.cumsum(EddStEx, dim=1)  # sum of e_dd in C for denominator
        Edd_inMulSum = torch.cumsum(EddZinMul, dim=1)  # cumulative sum of e_dd*z_in for numerator
        z_outAll = Edd_inMulSum / torch.clamp(EddSum - self.th, 1e-10, 1e10)  # all assumed outputs
        z_outAll = torch.where(EddSum < self.th, 1e5 * torch.ones_like(z_outAll), z_outAll)  # exclude denominator = 0
        z_outAll = torch.where(z_outAll < z_inStEx, 1e5 * torch.ones_like(z_outAll), z_outAll)  # exclude in after out
        z_out, z_outInd = torch.min(z_outAll, dim=1)  # the first assumed output is the real output

        # Complete casual set record
        _, inStIndR = torch.sort(z_inStInd, dim=1)  # for reverse the order
        z_outIndEx = torch.tile(torch.reshape(z_outInd, [bs, 1, outCh]), [1, inCh, 1])
        locEx = torch.tile(torch.reshape(torch.arange(inCh), [1, inCh, 1]), [bs, 1, outCh]).to(dv)
        self.cause_mask = torch.where(locEx <= z_outIndEx, 1, 0)  # record of casual set after sorting
        self.cause_mask = torch.gather(self.cause_mask, dim=1, index=torch.tile(
            torch.reshape(inStIndR, [bs, self.inCh, 1]), [1, 1, self.outCh]))  # record of casual set before sorting

        return z_out * self.e_da

    def pass_delta(self, bs, delta):  # Supervised Term
        bs, inCh, outCh = bs, self.inCh, self.outCh  # batch size, channels
        e_dd2 = torch.tile(torch.reshape(self.e_dd, [1, inCh, outCh]), [bs, 1, 1])
        delta2Ex = torch.tile(torch.reshape(delta, [bs, 1, outCh]), [1, inCh, 1])
        delta_out = torch.sum(delta2Ex * self.cause_mask * e_dd2, dim=2)
        return delta_out

    def backward(self, bs, delta, z_in, z_out, lr_Edd, lr_Eda):  # DSNN-ETDP
        bs, inCh, outCh = bs, self.inCh, self.outCh  # batch size, channels
        e_daEx = torch.tile(torch.reshape(torch.exp(self.e_da), [1, 1, outCh]), [bs, inCh, 1])
        deltaEx = torch.tile(torch.reshape(delta, [bs, 1, outCh]), [1, inCh, 1])  # Broadcast Term
        z_inEx = torch.tile(torch.reshape(z_in, [bs, inCh, 1]), [1, 1, outCh])  # z_in
        z_outEx = torch.tile(torch.reshape(z_out, [bs, 1, outCh]), [1, inCh, 1])  # z_out
        T = (z_inEx - z_outEx) * self.cause_mask  # Local Term (exponential time difference)
        adj_Edd = torch.sum(deltaEx * e_daEx * T, dim=0)
        adj_Eda = torch.sum(delta * z_out, dim=0)

        # Adam
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
