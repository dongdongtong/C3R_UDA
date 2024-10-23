import torch
import torch.nn as nn


class Discriminator(nn.Module):

    # note that when using default patchGAN disciminator(1st conv k_s=4, pad=1), avg Dice could achieve 80+
    # when using (1st conv k_s=6, pad=2), avg Dice could achieve 79+
    # when using default patchGAN discriminator(but last conv k_s=4, stride=2), avg Dice could achieve 
    # when using 5 conv block patchGAN discriminator, avg Dice could achieve 
    def __init__(self, in_ch, norm_type="batch_norm"):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, kernel_size=4, stride=2, padding=1, normalize=True, norm_type="batch_norm", use_bias=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size, stride=stride, padding=padding, bias=use_bias)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters) if norm_type == "batch_norm" else nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_ch, 64, 4, padding=1, normalize=False, use_bias=True),
            *discriminator_block(64, 128, norm_type=norm_type, normalize=True, use_bias=True),
            *discriminator_block(128, 256, norm_type=norm_type, normalize=True, use_bias=True),
            *discriminator_block(256, 512, stride=2, norm_type=norm_type, normalize=True, use_bias=True),
            # *discriminator_block(512, 512, stride=1, norm_type=norm_type, normalize=True, use_bias=True),
            # nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, stride=1, padding=0, bias=True)
        )

    def forward(self, img):
        return self.model(img)


class DomainClassifier(nn.Module):
    def __init__(self, in_ch, norm_type="BatchNorm"):
        super(DomainClassifier, self).__init__()

        def discriminator_block(in_filters, out_filters, stride=2, normalize=True, norm_type="BatchNorm", use_bias=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=stride, padding=1, bias=use_bias)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters) if norm_type == "BatchNorm" else nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_ch, 64, normalize=False, use_bias=True),
            *discriminator_block(64, 128, norm_type=norm_type, normalize=True, use_bias=False),
            *discriminator_block(128, 256, norm_type=norm_type, normalize=True, use_bias=False),
            *discriminator_block(256, 512, stride=2, norm_type=norm_type, normalize=True, use_bias=False),
            # nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, stride=1, padding=1, bias=True)
        )

    def forward(self, img):
        return self.model(img)


class DomainClassifierFeatures(nn.Module):
    def __init__(self, in_features, nf):
        super(DomainClassifierFeatures, self).__init__()

        self.model = nn.Sequential(nn.Linear(in_features, nf),
                              nn.LeakyReLU(0.2, True),
                              nn.Linear(nf, nf),
                              nn.LeakyReLU(0.2, True),
                              nn.Linear(nf, nf),
                              nn.LeakyReLU(0.2, True),
                              nn.Linear(nf, 1))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)


if __name__ == "__main__":
    from torchsummary import summary
    dis = Discriminator(1, "InstanceNorm").cuda()

    summary(dis, (1,256,256))
