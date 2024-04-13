from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

"""Code taken from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
Altered for our purposes."""


def conv1d(in_planes: int, out_planes:int, kernel_size:int = 3, 
           stride: int = 1, dilation:int =1, bn:bool = True, drop_outP: float = 0.):
    layers = []
    if (drop_outP>0):
        layers.append(nn.Dropout(drop_outP))
    layers.append(nn.Conv1d(in_planes, out_planes, kernel_size, 
                            stride, padding = (kernel_size-1)//2, dilation = dilation, bias=not(bn)))
    if bn: layers.append(nn.BatchNorm1d(out_planes))
    return nn.Sequential(*layers) #*layers here means the sequential unpacking of the list

def FC_linear(in_planes:int, out_planes:int, bn:bool = True):
    layers = []
    if bn: list.append(nn.BatchNorm1d(in_planes))
    layers.append(nn.Linear(in_planes, out_planes)); layers.append(nn.ReLU())
    return nn.Sequential(*layers)  



class BasicConv1d(nn.Sequential):

    def __init__(self, input_channels: int = 8, filters:List[int] = [128, 128, 128, 128], kernel_size: int = 3, stride: int = 2, dilation : int = 1, 
                 pool_kernel: int = 1, pool_stride:int = 1, num_classes: int = 2, linear_layers:int = 2, bn: bool = True, drop_p: float = 0.):
        layers = []
        for i in range(len(filters)):
            inner_layer = []
            inner_layer.append(conv1d(in_planes = input_channels if i == 0 else filters[i-1], out_planes=filters[i], 
                                      kernel_size=kernel_size, stride=stride, dilation=dilation, bn=bn, drop_outP=drop_p))
            if pool_kernel > 0 and i < len(filters)-1:
                inner_layer.append(nn.MaxPool1d(pool_kernel, pool_stride, padding=(pool_kernel-1)//2))
            
            layers.append(nn.Sequential(*inner_layer))

        for i in range(linear_layers):
            inner_layer = []
            inner_layer.append()
        

            
            







class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self, inplanes: int, planes: int,  
        stride: int = 1,  downsample: Optional[nn.Module] = None,  groups: int = 1,
        base_width: int = 64, dilation: int = 1, norm_layer: Optional[Callable[..., nn.Module]] = None, ) -> None:
        
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x); out = self.relu(out); out = self.bn1(out); 
        out = self.conv2(out); out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv1d:
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv1d:
    """3x3 convolution with padding"""
    return nn.Conv1d( in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation,
        groups=groups, bias=False, dilation=dilation)

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self, inplanes: int, planes: int,
        stride: int = 1, downsample: Optional[nn.Module] = None, groups: int = 1, 
        base_width: int = 64, dilation: int = 1, norm_layer: Optional[Callable[..., nn.Module]] = None, ) -> None:
        
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x); out = self.relu(out); out = self.bn1(out)
        out = self.conv2(out); out = self.relu(out); out = self.bn2(out)
        out = self.conv3(out); out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__( self, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int],
        num_classes: int = 1000, zero_init_residual: bool = False, groups: int = 1, 
        width_per_group: int = 64, replace_stride_with_dilation: Optional[List[bool]] = None, norm_layer: Optional[Callable[..., nn.Module]] = None,) -> None:
        
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv1d(12, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer( self, block: Type[Union[BasicBlock, Bottleneck]],
                        planes: int, blocks: int, stride: int = 1,
                        dilate: bool = False, ) -> nn.Sequential:
        
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion), )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block( self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer,))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x); 
        x = self.relu(x); 
        x = self.bn1(x); 
        x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x); x = torch.flatten(x, 1); x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], progress: bool, **kwargs: Any,) -> ResNet:

    model = ResNet(block, layers, **kwargs)
    return model

def resnet101() -> ResNet:
    return _resnet(Bottleneck, [3,4,23,3], progress=False)