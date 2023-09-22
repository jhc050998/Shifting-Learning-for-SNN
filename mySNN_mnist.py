import math
import time
import torch

import Dataset as Ds
import DSNNLayer as mySNN


def train(sn, bs):
    gpu_num = torch.cuda.device_count()
    if gpu_num == 0:
        print("Training on cpu.")
        device = torch.device("cpu")
    else:
        print("Training on gpu.")
        device = torch.device("cuda:0")

    # Network layout
    ly1 = mySNN.DSNNLayer(inCh=784, outCh=800)
    ly2 = mySNN.DSNNLayer(inCh=800, outCh=10)

    # send parameters to device
    ly1.e_dd, ly1.e_da = ly1.e_dd.to(device), ly1.e_da.to(device)
    ly2.e_dd, ly2.e_da = ly2.e_dd.to(device), ly2.e_da.to(device)
    ly1.cause_mask = ly1.cause_mask.to(device)
    ly2.cause_mask = ly2.cause_mask.to(device)
    ly1.adam_m_Edd, ly1.adam_m_Eda = ly1.adam_m_Edd.to(device), ly1.adam_m_Eda.to(device)
    ly2.adam_m_Edd, ly2.adam_m_Eda = ly2.adam_m_Edd.to(device), ly2.adam_m_Eda.to(device)
    ly1.adam_v_Edd, ly1.adam_v_Eda = ly1.adam_v_Edd.to(device), ly1.adam_v_Eda.to(device)
    ly2.adam_v_Edd, ly2.adam_v_Eda = ly2.adam_v_Edd.to(device), ly2.adam_v_Eda.to(device)

    # Data prepare
    X_train, y_train = [], []
    for idx, (data, target) in enumerate(Ds.fashion_mnist_train_loader):  # read out all data in one time
        X_train, y_train = data, target
    X_train = torch.where(X_train > 0.5, 0.01, 2.3)
    X_train, y_train = X_train.to(device), y_train.to(device)

    # Training process
    epoch_num = 40
    lr_start, lr_end = 1e-4, 1e-6  # decaying learning rate for dendrite delays
    lr_decay = (lr_end / lr_start) ** (1.0 / epoch_num)
    lr_Eda = 1e-4  # learning rate for axon delays

    bn = int(math.ceil(sn / bs))
    loss, total_loss = 0, []
    time_start = time.time()  # time when training process start
    for epoch in range(epoch_num):  # 6000
        lr_Edd = lr_start * lr_decay ** epoch
        for bi in range(bn):  # 20
            # input data
            if (bi + 1) * bs > sn:
                data, tar = X_train[bi * bs:sn], y_train[bi * bs:sn]
            else:
                data, tar = X_train[bi * bs:(bi + 1) * bs], y_train[bi * bs:(bi + 1) * bs]
            z0 = torch.exp(1.0 - data.view(-1, 28 * 28))  # processing data (bs,1,28,28) --> (bs,784)
            tar_10 = (torch.ones(tar.size()[0], 10)*0.99).to(device)  # the prepared label
            for i in range(data.size()[0]):
                tar_10[i, tar[i]] = 0.01

            bs = z0.size()[0]

            # Forward propagation
            z1 = ly1.forward(bs, z0, dv=device)
            z2 = ly2.forward(bs, z1, dv=device)

            # Backward propagation
            z2_lo, z_tar = torch.softmax(z2, dim=1), torch.softmax(torch.exp(tar_10), dim=1)
            delta2 = z2_lo - z_tar
            delta1 = ly2.pass_delta(bs, delta2)
            ly2.backward(bs, delta2, z1, z2, lr_Edd, lr_Eda)
            ly1.backward(bs, delta1, z0, z1, lr_Edd, lr_Eda)

            CE = -1.0 * torch.sum(torch.log(torch.clamp(z2_lo, 1e-5, 1.0)) * z_tar) / data.size()[0]
            CE_min = -1.0 * torch.sum(torch.log(torch.clamp(z_tar, 1e-5, 1.0)) * z_tar) / data.size()[0]
            loss = abs(CE - CE_min)

            if bi % 10 == 0:
                print("Current Training epoch: " + str(epoch + 1), end="\t")
                print("Progress: [" + str(bi * bs) + "/" + str(sn), end="")
                print("(%.0f %%)]" % (100.0 * bi * bs / sn), end="\t")
                print("Error: " + str(loss))
                total_loss.append(loss)
        pass
        time_epoch_end = time.time()
        print("Time consuming: %.3f s" % (time_epoch_end - time_start))
        torch.save(ly1.e_dd, "./parameters_record/DSNN_mnist_edd1")
        torch.save(ly1.e_da, "./parameters_record/DSNN_mnist_eda1")
        torch.save(ly2.e_dd, "./parameters_record/DSNN_mnist_edd2")
        torch.save(ly2.e_da, "./parameters_record/DSNN_mnist_eda2")
        print("Accuracy on test data: ")
        test(10000, 100)
    pass
    time_end = time.time()  # time when training process end
    print("Time consuming: %.3f s" % (time_end - time_start))
    print(torch.tensor(total_loss).size())
    print(torch.tensor(total_loss))


def test(sn, bs):
    gpu_num = torch.cuda.device_count()
    if gpu_num == 0:
        print("Testing on cpu.")
        device = torch.device("cpu")
    else:
        print("Testing on gpu.")
        device = torch.device("cuda:0")

    e_dd1 = torch.load("./parameters_record/DSNN_mnist_edd1").to(device)
    e_da1 = torch.load("./parameters_record/DSNN_mnist_eda1").to(device)
    e_dd2 = torch.load("./parameters_record/DSNN_mnist_edd2").to(device)
    e_da2 = torch.load("./parameters_record/DSNN_mnist_eda2").to(device)

    ly1 = mySNN.DSNNLayer(784, 800, e_dd=e_dd1, e_da=e_da1)
    ly2 = mySNN.DSNNLayer(800, 10, e_dd=e_dd2, e_da=e_da2)

    # Data prepare
    X_test, y_test = [], []
    for idx, (data, target) in enumerate(Ds.fashion_mnist_test_loader):
        X_test, y_test = data, target
    X_test = torch.where(X_test > 0.5, 0.01, 2.3)
    X_test, y_test = X_test.to(device), y_test.to(device)

    # Testing Process
    correct = 0
    bn = int(math.ceil(sn / bs))
    for bi in range(bn):
        if (bi + 1) * bs > sn:
            data, tar = X_test[bi * bs:sn], y_test[bi * bs:sn]
        else:
            data, tar = X_test[bi * bs:(bi + 1) * bs], y_test[bi * bs:(bi + 1) * bs]
        z0 = torch.exp(1.0 - data.view(-1, 28 * 28))

        # Forward propagation
        z1 = ly1.forward(bs, z0, dv=device)
        z2 = ly2.forward(bs, z1, dv=device)

        lo = torch.softmax(z2, dim=1)
        prediction = torch.argmin(lo, dim=1)
        correct += prediction.eq(tar.data).sum()
    pass
    print("Accuracy: " + str(int(correct)) + "/" + str(sn), end="")
    print("(%.3f %%)" % (100. * correct / sn))


def main():
    train(60000, 128)
    test(10000, 100)


if __name__ == "__main__":
    main()
