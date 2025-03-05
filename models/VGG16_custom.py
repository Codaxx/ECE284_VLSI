
import torch
import torch.nn as nn
import math
from models.quant_layer import *

cfg = {
    # 'VGG16_quant': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG16_custom': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 8, 'C', 512, 'M'],
}

# 512 => 8
# C: 8 => 8
# 8 => 512 

class VGG_quant(nn.Module):
    def __init__(self, vgg_name):
        super(VGG_quant, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif x == 'C': # squeezed convolution for 2D systolic array
                layers += [QuantConv2d(in_channels, 8, kernel_size=3, padding=1),
                           nn.ReLU(inplace=True)]
                in_channels = 8
            else:
                layers += [QuantConv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantConv2d):
                m.show_params()
    

def VGG16_custom(**kwargs):
    model = VGG_quant(vgg_name = 'VGG16_custom', **kwargs)
    return model



