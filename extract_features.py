import torch
import torchvision
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from PIL import Image
from torch.autograd import Variable
from torchvision import models, transforms


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers=[]):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    # 自己修改forward函数
    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name == 'module':
                for lname, layer in module._modules.items():
                    if lname is "fc": x = x.view(x.size(0), -1)
                    x = layer(x)
                    outputs.append(x)
            else:
                print("Error in extracting")
                exit(-1)
        return outputs