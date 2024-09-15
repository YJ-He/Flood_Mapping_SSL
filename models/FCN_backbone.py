import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.vgg import VGG
from models.ASPP import build_aspp

ranges = {
    'vgg11': ((0, 3), (3, 6), (6, 11), (11, 16), (16, 21)),
    'vgg11_bn': ((0, 4), (4, 8), (8, 15), (15, 22), (22, 29)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg13_bn': ((0, 7), (7, 14), (14, 21), (21, 28), (28, 35)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg16_bn': ((0, 7), (7, 14), (14, 24), (24, 34), (34, 44)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37)),
    'vgg19_bn': ((0, 7), (7, 14), (14, 27), (27, 40), (40, 53))
}

cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def conv_relu_bn(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=True),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels),
    )

class FCNs_VGG_ASPP_4conv1(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, backbone='vgg16_bn', pretrained=True, requires_grad=True, remove_fc=True):
        super().__init__()
        self.name = "FCNs_BCELoss_" + backbone + '_ASPP_4conv1'

        assert backbone in ['vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn']
        self.pretrained_net = VGGNet(pretrained=pretrained, model=backbone, requires_grad=requires_grad,
                                     remove_fc=remove_fc,
                                     batch_norm='bn' in backbone)
        # input channel!=3
        if in_ch != 3:
            self.pretrained_net.features[0] = nn.Conv2d(in_ch, 64, 3, 1, 1)

        self.aspp = build_aspp(backbone='vgg', output_stride=16, BatchNorm=nn.BatchNorm2d)
        self.conv4 = conv_relu_bn(512, 256)
        self.conv3 = conv_relu_bn(256, 128)
        self.conv2 = conv_relu_bn(128, 64)
        self.conv1 = conv_relu_bn(64, 32)

        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512 + 256, 256, kernel_size=3, stride=2, padding=1, dilation=1,
                                          output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256 + 128, 128, kernel_size=3, stride=2, padding=1, dilation=1,
                                          output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128 + 64, 64, kernel_size=3, stride=2, padding=1, dilation=1,
                                          output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64 + 32, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier1 = nn.Conv2d(32, out_ch, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']
        x4 = output['x4']
        x3 = output['x3']
        x2 = output['x2']
        x1 = output['x1']

        x5 = self.aspp(x5)
        x4 = self.conv4(x4)
        x3 = self.conv3(x3)
        x2 = self.conv2(x2)
        x1 = self.conv1(x1)

        score = self.bn1(self.relu(self.deconv1(x5)))
        score = torch.cat([score, x4], dim=1)
        score = self.bn2(self.relu(self.deconv2(score)))
        score = torch.cat([score, x3], dim=1)
        score = self.bn3(self.relu(self.deconv3(score)))
        score = torch.cat([score, x2], dim=1)
        score = self.bn4(self.relu(self.deconv4(score)))
        score = torch.cat([score, x1], dim=1)
        score = self.bn5(self.relu(self.deconv5(score)))
        score = self.classifier1(score)

        return score

class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False,
                 batch_norm=False):
        super().__init__(make_layers(cfg[model.replace('_bn', '')], batch_norm))
        self.ranges = ranges[model]

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant params
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}
        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d" % (idx + 1)] = x

        return output


if __name__ == "__main__":
    model = FCNs_VGG_ASPP_4conv1(in_ch=3, out_ch=1, backbone="vgg16_bn")
    print(model)


