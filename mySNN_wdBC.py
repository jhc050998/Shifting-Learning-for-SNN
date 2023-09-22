import math
import time
import torch

import Dataset as Ds
import DSNNLayer as mySNN


def wdBC():
    # Data prepare
    X, y = Ds.wdBC_loader()  # X for inputs, y for label
    X, y = X[0:560], y[0:560]

    time_start = time.time()  # time when training process start
    total_correct = 0
    for f in range(5):  # 5-fold cross validation
        print("Fold: " + str(f+1) + ".")

        # Network layout
        ly1 = mySNN.DSNNLayer(inCh=30, outCh=40)
        ly2 = mySNN.DSNNLayer(inCh=40, outCh=2)

        # Read out data
        X_train, y_train = torch.cat(
            (X[0:f*112], X[f*112+112:1120]), dim=0), torch.cat((y[0:f*112], y[f*112+112:1120]), dim=0)
        X_test, y_test = X[f*112:f*112+112], y[f*112:f*112+112]

        # Training process
        epoch_num = 400
        lr_start, lr_end = 1e-2, 1e-3  # decaying learning rate for dendrite delays
        lr_decay = (lr_end / lr_start) ** (1.0 / epoch_num)
        lr_Eda = 1e-4  # learning rate for axon delays

        sn, bs = 448, 112  # number of data samples, batch size
        bn = int(math.ceil(sn / bs))  # number of batches
        loss, total_loss = 0, []
        for epoch in range(epoch_num):
            lr_Edd = lr_start * lr_decay ** epoch
            for bi in range(bn):
                # input data
                if (bi + 1) * bs > sn:  # for the last batch with unusual size
                    data, tar_2 = X_train[bi * bs:sn], y_train[bi * bs:sn]
                else:  # for other batches
                    data, tar_2 = X_train[bi * bs:(bi + 1) * bs], y_train[bi * bs:(bi + 1) * bs]
                z0 = torch.exp(1.0 - data)  # (bs=56, 30) temporal-coding

                # Forward propagation
                z1 = ly1.forward(bs, z0)
                z2 = ly2.forward(bs, z1)

                # Backward propagation
                z2_lo, z_tar = torch.softmax(z2, dim=1), torch.softmax(torch.exp(1.0 - tar_2), dim=1)
                delta2 = z2_lo - z_tar
                delta1 = ly2.pass_delta(bs, delta2)
                ly2.backward(bs, delta2, z1, z2, lr_Edd, lr_Eda)
                ly1.backward(bs, delta1, z0, z1, lr_Edd, lr_Eda)

                CE = -1.0 * torch.sum(torch.log(torch.clamp(z2_lo, 1e-5, 1.0)) * z_tar) / data.size()[0]
                CE_min = -1.0 * torch.sum(torch.log(torch.clamp(z_tar, 1e-5, 1.0)) * z_tar) / data.size()[0]
                loss = abs(CE - CE_min)
            pass
            if epoch % 50 == 0:
                print("Current Training epoch: " + str(epoch + 1), end="\t")
                print("Progress: [" + str(epoch) + "/" + str(epoch_num), end="")
                print("(%.0f %%)]" % (100.0 * epoch / epoch_num), end="\t")
                print(" ")
                print("Error: " + str(loss))
                total_loss.append(loss)
        pass

        # total_loss = torch.tensor(total_loss)
        # print(total_loss)

        # Testing Process
        correct = 0
        sn, bs = 112, 112  # number of data samples, batch size
        bn = int(math.ceil(sn / bs))  # number of batches
        for bi in range(bn):
            if (bi + 1) * bs > sn:
                data, tar_3 = X_test[bi * bs:sn], y_test[bi * bs:sn]
            else:
                data, tar_3 = X_test[bi * bs:(bi + 1) * bs], y_test[bi * bs:(bi + 1) * bs]
            z0 = torch.exp(1.0 - data)  # (bs=56, 30)
            tar = torch.argmax(tar_3, dim=1)

            # Forward propagation
            z1 = ly1.forward(bs, z0)
            z2 = ly2.forward(bs, z1)

            lo = z2
            prediction = torch.argmin(lo, dim=1)
            correct += prediction.eq(tar.data).sum()
        pass
        total_correct += int(correct)
        print("Accuracy: " + str(int(correct)) + "/" + str(sn), end="")
        print("(%.3f %%)" % (100. * correct / sn))
        print(" ")
    pass
    time_end = time.time()  # time when training process end

    print(" ")
    print("5-fold accuracy: " + str(total_correct) + "/" + str(560), end="")
    print("(%.3f %%)" % (100. * total_correct / 560))
    print("Time consuming: %.3f s" % (time_end - time_start))


def main():
    wdBC()


if __name__ == "__main__":
    main()
