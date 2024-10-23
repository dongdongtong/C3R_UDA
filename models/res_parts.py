import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, dilation=1, norm="BatchNorm", act="leaky_relu"):
        super(ResBlock, self).__init__()

        self.diff = (in_ch != out_ch) or stride == 2

        if norm == "BatchNorm":
            self.norm = nn.BatchNorm2d
            self.use_bias = False
        elif norm == "InstanceNorm":
            self.norm = nn.InstanceNorm2d
            self.use_bias = True
        # elif norm == "SyncronizedBatchnorm":
        #     self.norm = SynchronizedBatchNorm2d

        if self.diff:
            self.adjust_conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=self.use_bias),
                self.norm(out_ch)
            )

        self.mpconv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=self.use_bias),
            self.norm(out_ch),
            nn.LeakyReLU(0.2, inplace=True) if act == "leaky_relu" else nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, kernel_size, stride=1, padding=dilation, dilation=dilation, bias=self.use_bias),
            self.norm(out_ch)
        )
        self.lrelu = nn.LeakyReLU(0.2, inplace=True) if act == "leaky_relu" else nn.ReLU(True)

    def forward(self, x):
        x_s = self.shortcut(x)

        x = self.mpconv(x)
        return self.lrelu(x_s + x)
    
    def shortcut(self, x):
        if self.diff:
            x_s = self.adjust_conv(x)
        else:
            x_s = x

        return x_s



class ResDown(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, blocks=2, norm="BatchNorm", act="leaky_relu"):
        super(ResDown, self).__init__()
        main = [ResBlock(in_ch, out_ch, kernel_size, stride=2, norm=norm, act=act)]

        for i in range(blocks - 1):
            main.append(ResBlock(out_ch, out_ch, kernel_size, stride=1, norm=norm, act=act))
        
        self.model = nn.Sequential(*main)

    def forward(self, x):
        out = self.model(x)

        return out


class ResUp(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, nearest=True, norm="BatchNorm", up=True, act="leaky_relu"):
        super(ResUp, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights

        if norm == "BatchNorm":
            self.norm = nn.BatchNorm2d
            self.use_bias = False
        elif norm == "InstanceNorm":
            self.norm = nn.InstanceNorm2d
            self.use_bias = True

        self.up = up

        if self.up:
            if nearest:
                self.up_conv = nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=self.use_bias),
                    self.norm(out_ch),
                    nn.LeakyReLU(0.2, inplace=True) if act == "leaky_relu" else nn.ReLU(True),
                )
            else:
                self.up_conv = nn.Sequential(
                    nn.ConvTranspose2d(out_ch, out_ch, 3, stride=2, padding=1, bias=self.use_bias),
                    nn.ConstantPad2d((0, 1, 0, 1), value=0),
                    self.norm(out_ch),
                    nn.LeakyReLU(0.2, inplace=True) if act == "leaky_relu" else nn.ReLU(True),
                    nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=self.use_bias),
                    self.norm(out_ch),
                    nn.LeakyReLU(0.2, inplace=True) if act == "leaky_relu" else nn.ReLU(True),
                )

        self.res = ResBlock(in_ch, out_ch, 3, 1, norm=norm, act=act)

    def forward(self, x):
        o_res = self.res(x)

        if self.up:
            o_up = self.up_conv(o_res)
        else:
            o_up = o_res

        return o_up