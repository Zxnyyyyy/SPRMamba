import torch
import torch.nn as nn
from torchvision import models, transforms
from thop import profile

class resnet50(torch.nn.Module):
    def __init__(self, num_class):
        super(resnet50, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)
        self.share.add_module("avgpool", resnet.avgpool)
        self.fc = nn.Sequential(nn.Linear(2048, 512),
                                nn.ReLU(),
                                nn.Linear(512, num_class))

    def forward_features(self, x):
        x = self.share.forward(x)
        x = x.view(-1, 2048)
        return x

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = self.forward_features(x)
        y = self.fc(x)
        return y

if __name__ == '__main__':
    num_gpu = torch.cuda.device_count()
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(0) if use_gpu else "cpu")
    model = resnet50(num_class=8).to(device)
    # dummy_input = torch.randn(1, 3, 224, 224).to(device)
    # flops, params = profile(model, (dummy_input,))
    # print('flops: ', flops, 'params: ', params)
    # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    a = model.forward_features(dummy_input)
    print(a.shape)