import torch
import torch.nn as nn


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, kernel_size=3, norm="BatchNorm", act="leaky_relu"):
        super(double_conv, self).__init__()
        if norm == "BatchNorm":
            norm_layer = nn.BatchNorm2d
        elif norm == "InstanceNorm":
            norm_layer = nn.InstanceNorm2d
        # elif norm == "SyncronizedBatchnorm":
        #     norm_layer = SynchronizedBatchNorm2d

        if act == "leaky_relu":
            self.act = nn.LeakyReLU(0.2, True)
        elif act == "relu":
            self.act = nn.ReLU(True)
        

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size,
                      padding=(kernel_size//2, kernel_size-1-kernel_size//2)),
            norm_layer(out_ch),
            self.act,
            # nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size,
                      padding=(kernel_size//2, kernel_size-1-kernel_size//2)),
            norm_layer(out_ch),
            # nn.InstanceNorm2d(out_ch),
            self.act,
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, norm="BatchNorm", act="leaky_relu"):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, kernel_size, norm=norm, act=act)

    def forward(self, x):
        x = self.conv(x)
        return x
        

class down(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, norm="BatchNorm", act="leaky_relu"):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch, kernel_size, norm=norm, act=act)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x
        
        
class up(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, nearest=True, norm="BatchNorm", act="leaky_relu"):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if nearest:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            prev_ch = in_ch - out_ch
            self.up = nn.ConvTranspose2d(prev_ch, prev_ch, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch, kernel_size, norm=norm, act=act)

    def forward(self, x1, x2, use_cat=True):
        # print("x1 shape:", x1.shape)
        x1 = self.up(x1)
        # print("x1 up shape:", x1.shape)
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        if use_cat:
            x = torch.cat([x1, x2], dim=1)
        else:
            x = x1
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x