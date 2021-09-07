import torch.nn as nn

from ..utils import Module


vgg_cfg = {
    'VGG11': [64, 'M',
              128, 'M',
              256, 'D', 256, 'M',
              512, 'D', 512, 'M',
              512, 'D', 512, 'M'],
    'VGG13': [64, 'D', 64, 'M',
              128, 'D', 128, 'M',
              256, 'D', 256, 'M',
              512, 'D', 512, 'M',
              512, 'D', 512, 'M'],
    'VGG16': [64, 'D', 64, 'M',
              128, 'D', 128, 'M',
              256, 'D', 256, 'D', 256, 'M',
              512, 'D', 512, 'D', 512, 'M',
              512, 'D', 512, 'D', 512, 'M'],
    'VGG19': [64, 'D', 64, 'M',
              128, 'D', 128, 'M',
              256, 'D', 256, 'D', 256, 'D', 256, 'M',
              512, 'D', 512, 'D', 512, 'D', 512, 'M',
              512, 'D', 512, 'D', 512, 'D', 512, 'M'],
}


class VGG(Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(vgg_cfg[vgg_name])
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
            elif x == 'D':
                layers += [nn.Dropout(0.)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.Dropout(0.)]
        return nn.Sequential(*layers)
