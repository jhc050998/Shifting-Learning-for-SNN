import time
import torch

import SNNLayer as mySNN


def Xor():
    # Network layout
    ly1 = mySNN.SNNLayer(inCh=2, outCh=8)
    ly2 = mySNN.SNNLayer(inCh=8, outCh=2)

    # Data prepare
    net_in = torch.tensor([[0.01, 0.01], [0.01, 0.99], [0.99, 0.01], [0.99, 0.99]])  # data (bs=4, 2)
    tar = torch.tensor([[0], [1], [1], [0]])  # label (bs=4, 1)
    z0 = torch.exp(1.0 - net_in)  # temporal-coding for inputs
    tar_2 = torch.ones(tar.size()[0], 2) * 0.99
    for i in range(z0.size()[0]):
        tar_2[i, tar[i]] = 0.01

    lr_Ets, lr_Etp = 1e-2, 1e-4  # learning rate for shifting timings
    bs = 4
    res = []
    for epoch in range(400):
        # Forward propagation
        z1 = ly1.forward(bs, z0)  # (bs=4, 8)
        z2 = ly2.forward(bs, z1)

        # Shifting Learning
        z2_lo, z_tar = torch.softmax(z2, dim=1), torch.softmax(torch.exp(tar_2), dim=1)
        delta2 = z2_lo - z_tar
        delta1 = ly2.pass_delta(bs, delta2)

        ly2.backward(bs, delta2, z1, z2, lr_Ets, lr_Etp)
        ly1.backward(bs, delta1, z0, z1, lr_Ets, lr_Etp)

        # print("Results for Xor：")
        # print("Label: " + str(torch.reshape(tar, [4])))
        res = torch.argmin(z2, dim=1)
        # print("Output: " + str(result))
        # print(" ")

    # print(ly1.e_da)
    # print(ly2.e_da)

    if res.equal(torch.reshape(tar, [4])):
        return 1
    else:
        return 0


def Xor_mly():
    # Network layout
    ly1 = mySNN.SNNLayer(inCh=2, outCh=8)
    ly2 = mySNN.SNNLayer(inCh=8, outCh=8)
    ly3 = mySNN.SNNLayer(inCh=8, outCh=2)

    # Data prepare
    net_in = torch.tensor([[0.01, 0.01], [0.01, 0.99], [0.99, 0.01], [0.99, 0.99]])  # data (bs=4, 2)
    tar = torch.tensor([[0], [1], [1], [0]])  # label (bs=4, 1)
    z0 = torch.exp(1.0 - net_in)  # temporal-coding for inputs
    tar_2 = torch.ones(tar.size()[0], 2) * 0.99
    for i in range(z0.size()[0]):
        tar_2[i, tar[i]] = 0.01

    lr_Ets, lr_Etp = 1e-2, 1e-3  # learning rate for shifting timings
    bs = 4
    res = []
    for epoch in range(300):
        # Forward propagation
        z1 = ly1.forward(bs, z0)  # (bs=4, 4)
        z2 = ly2.forward(bs, z1)
        z3 = ly3.forward(bs, z2)

        # Shifting Learning
        z3_lo, z_tar = torch.softmax(z3, dim=1), torch.softmax(torch.exp(tar_2), dim=1)
        delta3 = z3_lo - z_tar
        delta2 = ly3.pass_delta(bs, delta3)
        delta1 = ly2.pass_delta(bs, delta2)

        ly3.backward(bs, delta3, z2, z3, lr_Ets, lr_Etp)
        ly2.backward(bs, delta2, z1, z2, lr_Ets, lr_Etp)
        ly1.backward(bs, delta1, z0, z1, lr_Ets, lr_Etp)

        # print("Results for Xor：")
        # print("Label: " + str(torch.reshape(tar, [4])))
        res = torch.argmin(z3, dim=1)
        # print("Output: " + str(result))
        # print(" ")

    if res.equal(torch.reshape(tar, [4])):
        return 1
    else:
        return 0


def main():
    # Xor()

    correct = 0
    total_num = 100
    time_start = time.time()
    for i in range(total_num):
        if i % int(total_num/10) == 0:
            print("Current trial index: " + str(i+1) + ".")
        correct += Xor()
    time_end = time.time()

    print(" ")
    print("Accuracy: " + str(int(correct)) + "/" + str(total_num), end="")
    print("(%.3f %%)" % (100. * correct / total_num))
    print("Time consuming: %.3f s" % (time_end - time_start))


if __name__ == "__main__":
    main()
