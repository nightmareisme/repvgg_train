import torch
import  torchvision
from torchvision import transforms
import torch.nn as nn
import  torch.nn.functional as F
import torch.optim as optim
from  torch.optim import optimizer
import  os
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

device = torch.device("cuda : 0" if torch.cuda.is_available() else "cpu")
epochs = 20
acc = []
loss_num = []

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False,
                                        transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, num_workers=0)


testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=False,
                                       transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def Conv1x1BN(in_channels,out_channels, stride=1, groups=1, bias=False):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels)
        )

def Conv3x3BN(in_channels,out_channels, stride=1, groups=1, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=stride,padding=1, groups=groups, bias=bias),
        nn.BatchNorm2d(out_channels)
    )


class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1, deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy

        if self.deploy:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=3, stride=stride, padding=1, dilation=1, groups=groups, bias=True)
        else:
            self.conv1 = Conv3x3BN(in_channels, out_channels, stride=stride, groups=groups, bias=False)
            self.conv2 = Conv1x1BN(in_channels, out_channels, stride=stride, groups=groups, bias=False)

            self.identity = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.deploy:
            return self.act(self.conv(x))
        if self.identity is None:
            return self.act(self.conv1(x) + self.conv2(x))
        else:
            return self.act(self.conv1(x) + self.conv2(x) + self.identity(x))


class RepVGG(nn.Module):
    def __init__(self, block_nums, width_multiplier=None, group=1, num_classes=1000, deploy=False):
        super(RepVGG, self).__init__()
        self.deploy = deploy
        self.group = group
        assert len(width_multiplier) == 4

        self.stage0 = RepVGGBlock(in_channels=3,out_channels=min(64, int(64 * width_multiplier[0])), stride=2, deploy=self.deploy)
        self.cur_layer_idx = 1
        self.stage1 = self._make_layers(in_channels=min(64, int(64 * width_multiplier[0])), out_channels= int(64 * width_multiplier[0]), stride=2, block_num=block_nums[0])
        self.stage2 = self._make_layers(in_channels=int(64 * width_multiplier[0]), out_channels=int(128 * width_multiplier[1]), stride=2, block_num=block_nums[1])
        self.stage3 = self._make_layers(in_channels=int(128 * width_multiplier[1]), out_channels=int(256 * width_multiplier[2]), stride=2, block_num=block_nums[2])
        self.stage4 = self._make_layers(in_channels=int(256 * width_multiplier[2]), out_channels=int(512 * width_multiplier[3]), stride=2, block_num=block_nums[3])
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)

        self._init_params()

    def _make_layers(self, in_channels, out_channels, stride, block_num):
        layers = []
        layers.append(RepVGGBlock(in_channels,out_channels, stride=stride, groups=self.group if self.cur_layer_idx%2==0 else 1, deploy=self.deploy))
        self.cur_layer_idx += 1
        for i in range(block_num):
            layers.append(RepVGGBlock(out_channels,out_channels, stride=1, groups=self.group if self.cur_layer_idx%2==0 else 1, deploy=self.deploy))
            self.cur_layer_idx += 1
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        return out

def RepVGG_A0(deploy=False):
    return RepVGG(block_nums=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], group=1, deploy=deploy)

def RepVGG_A1(deploy=False):
    return RepVGG(block_nums=[2, 4, 14, 1], num_classes=10,
                  width_multiplier=[1, 1, 1, 2.5], group=1, deploy=deploy)

def RepVGG_A2(deploy=False):
    return RepVGG(block_nums=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1.5, 1.5, 1.5, 2.75], group=1, deploy=deploy)

def RepVGG_B0(deploy=False):
    return RepVGG(block_nums=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], group=1, deploy=deploy)

def RepVGG_B1(deploy=False):
    return RepVGG(block_nums=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], group=1, deploy=deploy)

def RepVGG_B1g2(deploy=False):
    return RepVGG(block_nums=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], group=2, deploy=deploy)

def RepVGG_B1g4(deploy=False):
    return RepVGG(block_nums=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], group=4, deploy=deploy)


def RepVGG_B2(deploy=False):
    return RepVGG(block_nums=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], group=1, deploy=deploy)

def RepVGG_B2g2(deploy=False):
    return RepVGG(block_nums=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], group=2, deploy=deploy)

def RepVGG_B2g4(deploy=False):
    return RepVGG(block_nums=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], group=4, deploy=deploy)

def RepVGG_B3(deploy=False):
    return RepVGG(block_nums=[1, 4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], group=1, deploy=deploy)

def RepVGG_B3g2(deploy=False):
    return RepVGG(block_nums=[1, 4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], group=2, deploy=deploy)

def RepVGG_B3g4(deploy=False):
    return RepVGG(block_nums=[1, 4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], group=4, deploy=deploy)

def train():

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        corret =0
        total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device) , labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            _,predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            corret +=(predicted == labels).sum().item()
            # print(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print("epoch : %d/1000" % epoch)
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                acc.append(corret / total)
                loss_num.append(running_loss / 100)
                running_loss = 0.0
                print("------------")
                print('Ac: %d %%' %(100 * corret / total ))

if __name__ == '__main__':

    net = RepVGG_A1()
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    train()

    f = open("loss.txt" , 'a')
    loss_num =str(loss_num)
    f.write(loss_num)

    print("Finished Training")











# print(net)
# input = torch.randn(10,3,32,32)
# out = net(input)
# print(out.shape)
