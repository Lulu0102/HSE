import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision.models import ResNet50_Weights
from torchvision.models import ResNet101_Weights
import numpy as np
import math,copy
            
class ResNet(nn.Module):

    def __init__(self, features_base, features_hier_list, fc_indim=512*4, num_classes=[1000,1000,1000,1000]):
        super(ResNet, self).__init__()
        
        self.features_base = features_base
        
        self.features_1 = features_hier_list[0]
        self.features_2 = features_hier_list[1]
        self.features_3 = features_hier_list[2]
        self.features_4 = features_hier_list[3]
        
        self.fc_1 = nn.Linear(fc_indim, num_classes[0])
        self.fc_2 = nn.Linear(fc_indim, num_classes[1])
        self.fc_3 = nn.Linear(fc_indim, num_classes[2])
        self.fc_4 = nn.Linear(fc_indim, num_classes[3])
        
        self.averpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        b,c,w,h = x.size()
        
        ##################################################
        x_base = self.features_base(x)
        
        #coarse branch based on the softmax+softmin loss
        x_1 = self.features_1(x_base)
        x_1 = self.averpool(x_1).view(b, -1)
        x_1_out = self.fc_1(x_1)
        
        x_2 = self.features_2(x_base)
        x_2 = self.averpool(x_2).view(b, -1)
        x_2_out = self.fc_2(x_2)
        
        x_3 = self.features_3(x_base)
        x_3 = self.averpool(x_3).view(b, -1)
        x_3_out = self.fc_3(x_3)
        
        x_4 = self.features_4(x_base)
        x_4 = self.averpool(x_4).view(b, -1)
        x_4_out = self.fc_4(x_4)

        return x_1_out,x_2_out,x_3_out,x_4_out
        
#########################################################################################
#########################################################################################
    
def resnet50(pretrained=True, num_classes=[], **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    features_tmp_0 = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
    features_base = torch.nn.Sequential(*list(features_tmp_0.children())[:-3])#conv1-3
    
    features_tmp_1 = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
    features_1 = torch.nn.Sequential(*list(features_tmp_1.children())[-3:-2])#conv1-3
    
    features_tmp_2 = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
    features_2 = torch.nn.Sequential(*list(features_tmp_2.children())[-3:-2])#conv1-3
    
    features_tmp_3 = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
    features_3 = torch.nn.Sequential(*list(features_tmp_3.children())[-3:-2])#conv1-3
    
    features_tmp_4 = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
    features_4 = torch.nn.Sequential(*list(features_tmp_4.children())[-3:-2])#conv1-3
    
    features_hier_list = [features_1,features_2,features_3,features_4]
    
    model = ResNet(features_base, features_hier_list, 512*4, num_classes, **kwargs)
    
    if pretrained:
        print('Load pre-trained Resnet50 suucess!')
    else:
        print('Load Resnet50 suucess!')
    
    return model
