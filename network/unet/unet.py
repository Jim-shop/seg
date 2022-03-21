import torch
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, in_channel, num_class) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, num_class, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_class),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_class, num_class, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_class)
        )

    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
    def __init__(self, in_channel, num_class) -> None:
        super().__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channel, num_class, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_class),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class UNet(nn.Module):
    def __init__(self, in_channel=3, num_class=4, pretrained_path=None) -> None:
        super().__init__()
        n1 = 16
        filters = [n1, n1*2, n1*4, n1*8, n1*16]

        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.cv1 = conv_block(in_channel, filters[0])
        self.cv2 = conv_block(filters[0], filters[1])
        self.cv3 = conv_block(filters[1], filters[2])
        self.cv4 = conv_block(filters[2], filters[3])
        self.cv5 = conv_block(filters[3], filters[4])

        self.up5 = up_conv(filters[4], filters[3])
        self.up4 = up_conv(filters[3], filters[2])
        self.up3 = up_conv(filters[2], filters[1])
        self.up2 = up_conv(filters[1], filters[0])

        self.upcv5 = conv_block(filters[4], filters[3])
        self.upcv4 = conv_block(filters[3], filters[2])
        self.upcv3 = conv_block(filters[2], filters[1])
        self.upcv2 = conv_block(filters[1], filters[0])

        self.cv = nn.Conv2d(filters[0], num_class,
                            kernel_size=3, stride=1, padding=1)
        self.active = torch.nn.Softmax(dim=0)

    def forward(self, x):
        e1 = self.cv1(x)
        e2 = self.cv2(self.mp1(e1))
        e3 = self.cv3(self.mp2(e2))
        e4 = self.cv4(self.mp3(e3))
        e5 = self.cv5(self.mp4(e4))

        d5 = self.up5(e5)
        d5 = self.upcv5(torch.cat((e4, d5), dim=1))

        d4 = self.up4(d5)
        d4 = self.upcv4(torch.cat((e3, d4), dim=1))

        d3 = self.up3(d4)
        d3 = self.upcv3(torch.cat((e2, d3), dim=1))

        d2 = self.up2(d3)
        d2 = self.upcv2(torch.cat((e1, d2), dim=1))

        return self.active(self.cv(d2))
