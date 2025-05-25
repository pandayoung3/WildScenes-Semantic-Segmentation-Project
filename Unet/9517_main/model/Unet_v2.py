import torch
import torch.nn as nn
import torchvision.models as models
from model.mobilenet_v2 import MobileNetV2Backbone


class MobileNetV2_UNet(nn.Module):
    '''
        mobilenet v2 + unet

    '''
    def __init__(self, classes=3,pretrain=True):

        super(MobileNetV2_UNet, self).__init__()
        # -----------------------------------------------------------------
        # encoder
        # ---------------------
        #self.feature = mobilenet_v2()
        self.feature = MobileNetV2Backbone(in_channels=3,pretrained=pretrain)

        # -----------------------------------------------------------------
        # decoder
        # ---------------------

        self.s5_up_conv = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                        nn.Conv2d(1280, 96, 3, 1, 1),
                                        nn.BatchNorm2d(96),
                                        nn.ReLU())
        self.s4_fusion = nn.Sequential(nn.Conv2d(96, 96, 3, 1, 1),
                                       nn.BatchNorm2d(96))

        self.s4_up_conv = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                        nn.Conv2d(96, 32, 3, 1, 1),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU())
        self.s3_fusion = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                       nn.BatchNorm2d(32))

        self.s3_up_conv = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                        nn.Conv2d(32, 24, 3, 1, 1),
                                        nn.BatchNorm2d(24),
                                        nn.ReLU())
        self.s2_fusion = nn.Sequential(nn.Conv2d(24, 24, 3, 1, 1),
                                       nn.BatchNorm2d(24))

        self.s2_up_conv = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                        nn.Conv2d(24, 16, 3, 1, 1),
                                        nn.BatchNorm2d(16),
                                        nn.ReLU())
        self.s1_fusion = nn.Sequential(nn.Conv2d(16, 16, 3, 1, 1),
                                       nn.BatchNorm2d(16))

        self.last_conv = nn.Conv2d(16, classes, 3, 1, 1)
        self.last_up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, input):

        # -----------------------------------------------
        # encoder
        # ---------------------
        s1, s2, s3, s4, s5 = self.feature(input)
        # -----------------------------------------------
        # decoder
        # ---------------------
        s4_ = self.s5_up_conv(s5)
        s4_ = s4_ + s4
        s4 = self.s4_fusion(s4_)

        s3_ = self.s4_up_conv(s4)
        s3_ = s3_ + s3
        s3 = self.s3_fusion(s3_)

        s2_ = self.s3_up_conv(s3)
        s2_ = s2_ + s2
        s2 = self.s2_fusion(s2_)

        s1_ = self.s2_up_conv(s2)
        s1_ = s1_ + s1
        s1 = self.s1_fusion(s1_)

        out = self.last_up(self.last_conv(s1))

        return out

    def _initialize_weights(self, *stages):
        for modules in stages:
            for module in modules.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()

    def freeze_backbone(self):
        for param in self.feature.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):

        for param in self.feature.parameters():
            param.requires_grad = True
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad=True
# Example usage
# if __name__ == "__main__":
#     model = MobileNetV2_UNet(n_classes=21)  # Change n_classes to the number of classes in your segmentation task
#     input_tensor = torch.randn(1, 3, 256, 256)  # Example input tensor
#     output = model(input_tensor)
#     print(output.shape)
