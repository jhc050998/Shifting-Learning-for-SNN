import torch

import DSNNLayer as mySNN


def Xor():
    # Network layout
    ly1 = mySNN.DSNNLayer(inCh=2, outCh=8)
    ly2 = mySNN.DSNNLayer(inCh=8, outCh=2)

    # Data prepare
    net_in = torch.tensor([[0.01, 0.01], [0.01, 0.99], [0.99, 0.01], [0.99, 0.99]])  # data (bs=4, 2)
    tar = torch.tensor([[0], [1], [1], [0]])  # label (bs=4, 1)
    z0 = torch.exp(1.0 - net_in)  # temporal-coding for inputs
    tar_2 = torch.ones(tar.size()[0], 2) * 0.99
    for i in range(z0.size()[0]):
        tar_2[i, tar[i]] = 0.01

    lr_Edd, lr_Eda = 1e-2, 1e-3  # learning rate for dendrite and axon delays
    bs = 4
    res = []
    for epoch in range(200):
        # Forward propagation
        z1 = ly1.forward(bs, z0)  # (bs=4, 8)
        z2 = ly2.forward(bs, z1)

        # Backward propagation
        z2_lo, z_tar = torch.softmax(z2, dim=1), torch.softmax(torch.exp(tar_2), dim=1)
        delta2 = z2_lo - z_tar
        delta1 = ly2.pass_delta(bs, delta2)
        ly2.backward(bs, delta2, z1, z2, lr_Edd, lr_Eda)
        ly1.backward(bs, delta1, z0, z1, lr_Edd, lr_Eda)

        # print("Results for Xor：")
        # print("Label: " + str(torch.reshape(tar, [4])))
        res = torch.argmin(z2, dim=1)
        # print("Output: " + str(result))
        # print(" ")

    if res.equal(torch.reshape(tar, [4])):
        return 1
    else:
        return 0


def main():
    correct = 0
    total_num = 100
    for i in range(total_num):
        if i % int(total_num/10) == 0:
            print("Current trial index: " + str(i+1) + ".")
        correct += Xor()
    print("Accuracy: " + str(int(correct)) + "/" + str(total_num), end="")
    print("(%.3f %%)" % (100. * correct / total_num))
    # Xor()


if __name__ == "__main__":
    main()
