# camera-ready
import torch.nn as nn
import torch.nn.functional as f

from .resnet import ResNet18_OS8
from .aspp import ASPP
from .resnext import ResNeXt18_OS8



class DeepLabV3(nn.Module):
    def __init__(self, n_classes=2, n_channels=3):
        super(DeepLabV3, self).__init__()

        self.n_classes = n_classes
        self.n_channels = n_channels
        self.name = 'deeplabv3'

        self.resnet = ResNet18_OS8()

        self.aspp = ASPP(num_classes=self.n_classes)

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))
        h = x.size()[2]
        w = x.size()[3]

        feature_map = self.resnet(x)  # (shape: (batch_size, 512, h/8, w/8))
        output = self.aspp(feature_map)  # (shape: (batch_size, num_classes, h/16, w/16))

        output = f.upsample(output, size=(h, w), mode="bilinear")  # (shape: (batch_size, num_classes, h, w))

        return output


class XDeepLabV3(nn.Module):
    def __init__(self, n_classes=2, n_channels=3):
        super(XDeepLabV3, self).__init__()

        self.n_classes = n_classes
        self.n_channels = n_channels
        self.name = 'xdeeplabv3'

        self.resnet = ResNeXt18_OS8()  # Change the backbone

        self.aspp = ASPP(num_classes=self.n_classes)

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))
        h = x.size()[2]
        w = x.size()[3]

        feature_map = self.resnet(x)  # (shape: (batch_size, 512, h/8, w/8))
        output = self.aspp(feature_map)  # (shape: (batch_size, num_classes, h/16, w/16))

        output = f.upsample(output, size=(h, w), mode="bilinear")  # (shape: (batch_size, num_classes, h, w))

        return output


# if __name__ == '__main__':
#     from resnet import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet101_OS16, ResNet152_OS16, ResNet18_OS8, \
#         ResNet34_OS8
#     from aspp import ASPP, ASPP_Bottleneck
#     from resnext import ResNeXt18_OS8
#
#     net = DeepLabV3()
#     from thop import clever_format
#     from thop import profile
#
#     input = torch.randn(2, 3, 512, 512)
#     macs, params = profile(net, inputs=(input,))
#     macs, params = clever_format([macs, params], "%.3f")
#     print('MACs:', macs, ' Params:', params)
