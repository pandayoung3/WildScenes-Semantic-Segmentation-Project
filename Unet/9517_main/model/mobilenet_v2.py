import torch
import torch.nn as nn
import torchvision.models as models


class MobileNetV2Backbone(nn.Module):
    """ MobileNetV2 Backbone
    """

    def __init__(self, in_channels,pretrained=True):
        super(MobileNetV2Backbone, self).__init__()
        self.in_channels = in_channels

        self.model = models.mobilenet_v2(pretrained=pretrained)
        self.enc_channels = [16, 24, 32, 96, 1280]

    def forward(self, x):
        # x = reduce(lambda x, n: self.model.features[n](x), list(range(0, 2)), x)
        x = self.model.features[0](x)
        x = self.model.features[1](x)
        enc2x = x

        # x = reduce(lambda x, n: self.model.features[n](x), list(range(2, 4)), x)
        x = self.model.features[2](x)
        x = self.model.features[3](x)
        enc4x = x

        # x = reduce(lambda x, n: self.model.features[n](x), list(range(4, 7)), x)
        x = self.model.features[4](x)
        x = self.model.features[5](x)
        x = self.model.features[6](x)
        enc8x = x

        # x = reduce(lambda x, n: self.model.features[n](x), list(range(7, 14)), x)
        x = self.model.features[7](x)
        x = self.model.features[8](x)
        x = self.model.features[9](x)
        x = self.model.features[10](x)
        x = self.model.features[11](x)
        x = self.model.features[12](x)
        x = self.model.features[13](x)
        enc16x = x

        # x = reduce(lambda x, n: self.model.features[n](x), list(range(14, 19)), x)
        x = self.model.features[14](x)
        x = self.model.features[15](x)
        x = self.model.features[16](x)
        x = self.model.features[17](x)
        x = self.model.features[18](x)
        enc32x = x
        return [enc2x, enc4x, enc8x, enc16x, enc32x]

