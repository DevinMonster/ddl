import torch

from utils.loss import UnbiasedCrossEntropyLoss, UnbiasedKDLoss, MiBLoss


def test_uce():
    # x = torch.arange(36, dtype=torch.float32).reshape((1, 4, 3, 3))
    x = torch.randn((1, 4, 3, 3))
    print(x)
    print("-------------------------------------")
    y = torch.tensor([[[0, 0, 0],
                       [3, 3, 0],
                       [3, 3, 3]]])
    print(UnbiasedCrossEntropyLoss(num_old_classes=3)(x, y))


def test_ukd():
    # x = torch.arange(36, dtype=torch.float32).reshape((1, 4, 3, 3))
    x = torch.randn((1, 5, 3, 3))
    y = torch.randn((1, 4, 3, 3))
    print(UnbiasedKDLoss()(x, y))


def test_mib():
    x_new = torch.randn((1, 4, 3, 3))
    x_old = torch.randn((1, 3, 3, 3))
    y = torch.tensor([[[0, 0, 0],
                       [3, 3, 0],
                       [3, 3, 3]]])
    mib = MiBLoss()
    print(mib(x_old, x_new,y))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_uce()
    # test_ukd()
    # test_mib()
    pass
