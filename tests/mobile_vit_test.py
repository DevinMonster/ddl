import torch

from models.mobile_vit import MobileViTEncoder


def test_output():
    imgs = torch.randn((3, 3, 256, 256))
    encoder = MobileViTEncoder()
    print(encoder(imgs).shape)


if __name__ == '__main__':
    test_output()
