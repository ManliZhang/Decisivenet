'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import params

def activation(x, l,c, temperature):
    x = F.relu(x)
    use_l = False
    if l != -1:
        use_l = True
    if use_l:
        x_ = x.view(x.size(0),l,-1,x.size(2),x.size(3))
    else:
        x_ = x.view(x.size(0),-1,c,x.size(2),x.size(3))
    if temperature == 0:
        x_max = torch.max(x_,dim=1,keepdim=True)[0]
        x_max = x_max == x_
        out = x*x_max.view(x.size())
    else:
        x_softmax = torch.softmax(temperature*x_,dim=1).view(x.size()).detach()
        out = x*x_softmax
    return out

def nb_ops_func(out,layer,l,c):
    if l==0 and c==0:
        return out.nelement()*layer.weight.shape[1]*layer.weight.shape[2]*layer.weight.shape[3]/out.shape[0]
    if l != -1:
        return out.nelement()*layer.weight.shape[1]*layer.weight.shape[2]*layer.weight.shape[3]/(l*out.shape[0])
    return out.nelement()*c*layer.weight.shape[2]*layer.weight.shape[3]/out.shape[0]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # self.shortcut = nn.Sequential()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        if stride != 1 or in_planes != self.expansion*planes:
            self.sc_conv = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            self.sc_bn = nn.BatchNorm2d(self.expansion*planes)

    # def forward(self, x):
    #     out = F.relu(self.bn1(self.conv1(x)))
    #     out = self.bn2(self.conv2(out))
    #     out += self.shortcut(x)
    #     out = F.relu(out)
    #     return out

    def temp_forward(self, x, l,c, temp):
        out = self.conv1(x)
               
        if (params.first_block == 1):
            params.nb_ops += nb_ops_func(out,self.conv1,l,c) 
        else:
            params.nb_ops += nb_ops_func(out,self.conv1,0,0)

        out = activation(self.bn1(out), l,c, temp)
        out = self.bn2(self.conv2(out))
        params.nb_ops += nb_ops_func(out,self.conv2,l,c) 
        if self.stride != 1 or self.in_planes != self.expansion*self.planes:
            out += self.sc_bn(self.sc_conv(x))

            if (params.first_block == 1):
                params.nb_ops += nb_ops_func(out,self.sc_conv,l,c) 
            else:
                params.nb_ops += nb_ops_func(out,self.sc_conv,0,0) 
        else:
            out += x
        out = activation(out, l,c, temp)
        params.first_block = 1
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes
        #self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.sc_conv = nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False)
            self.sc_bn = nn.BatchNorm2d(self.expansion*planes)
            
    #def forward(self, x):
    #    out = F.relu(self.bn1(self.conv1(x)))
    #    out = F.relu(self.bn2(self.conv2(out)))
    #    out = self.bn3(self.conv3(out))
    #    out += self.shortcut(x)
    #    out = F.relu(out)
    #    return out

    def temp_forward(self, x, l,c, temp):
        out = self.conv1(x)
        if (params.first_block == 1):
            params.nb_ops += nb_ops_func(out,self.conv1,l,c) 
        else:
            params.nb_ops += nb_ops_func(out,self.conv1,0,0)
        out = activation(self.bn1(out),l,c, temp)
        out = self.conv2(out)
        params.nb_ops += nb_ops_func(out,self.conv2,l,c) 
        out = activation(self.bn2(out),l,c, temp)
        out = self.conv3(out)
        params.nb_ops += nb_ops_func(out,self.conv3,l,c) 
        out = self.bn3(out)
        if self.stride != 1 or self.in_planes != self.expansion*self.planes:
            
            out += self.sc_bn(self.sc_conv(x))
            if (params.first_block == 1):
                params.nb_ops += nb_ops_func(out,self.sc_conv,l,c) 
            else:
                params.nb_ops += nb_ops_func(out,self.sc_conv,0,0) 
        else:
            out+= x
        #out += self.shortcut(x)
        out = activation(out,l,c, temp)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, args, num_classes=10):
        super(ResNet, self).__init__()
        self.args = args
        self.in_planes = args.feature_maps
        fms = args.feature_maps
        self.length = len(num_blocks)
        self.conv1 = nn.Conv2d(3, fms, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(fms)
        self.layer1 = self._make_layer(block, fms, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2*fms, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4*fms, num_blocks[2], stride=2)
        if self.length > 3:
            self.layer4 = self._make_layer(block, 8*fms, num_blocks[3], stride=2)
            self.linear = nn.Linear(8*fms*block.expansion, num_classes)
        else:
            self.linear = nn.Linear(4*fms*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.ModuleList(layers)

    def temp_forward(self, x, l,c, temp):
        out = F.relu(self.bn1(self.conv1(x)))
        params.nb_ops = nb_ops_func(out,self.conv1,0,0) 
        params.first_block=0
        for layer in self.layer1:
            out = layer.temp_forward(out, l,c, temp)
        for layer in self.layer2:
            out = layer.temp_forward(out, l,c, temp)
        for layer in self.layer3:
            out = layer.temp_forward(out, l,c, temp)
        if self.length > 3:
            for layer in self.layer4:
                out = layer.temp_forward(out, l,c, temp)
        out = F.avg_pool2d(out, out.shape[2])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(args):
    return ResNet(BasicBlock, [2,2,2,2], args)

def ResNet9(args):
    return ResNet(BasicBlock, [1,1,1], args)

def ResNet20(args):
    return ResNet(BasicBlock, [3,3,3], args)

def ResNet34(args):
    return ResNet(BasicBlock, [3,4,6,3], args)

def ResNet50(args):
    return ResNet(Bottleneck, [3,4,6,3], args)

def ResNet101(args):
    return ResNet(Bottleneck, [3,4,23,3], args)

def ResNet152(args):
    return ResNet(Bottleneck, [3,8,36,3], args)


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
