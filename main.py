import torch
import torch.nn as nn

from torch.utils import checkpoint
from memonger import SublinearSequential

if __name__ == '__main__':
    data = torch.randn(1, 3, 5, 5)

    net = SublinearSequential(
        nn.Conv2d(3, 3, kernel_size=3, padding=1),
        nn.BatchNorm2d(3),
        nn.ReLU(),
        nn.Conv2d(3, 3, kernel_size=3, padding=1),
        nn.BatchNorm2d(3),
        nn.ReLU(),
        nn.Conv2d(3, 3, kernel_size=3, padding=1),
        nn.BatchNorm2d(3),
        nn.ReLU(),
    )

    res1 = net(data).sum()

    net.set_reforward(False)
    res2 = net(data).sum()

    net2 = nn.Sequential(
        *list(net.children())
    )
    res3 = net2(data).sum()

    print(res1.data, res2.data, res3.data)
