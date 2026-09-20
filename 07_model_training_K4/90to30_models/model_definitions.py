from __future__ import annotations
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


def frozen_resnet18_head(num_classes: int):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    for p in model.parameters(): p.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    # Backbone BatchNorm stays frozen even during model.train().
    def keep_frozen_bn_eval(module):
        for name, child in module.named_children():
            if name != 'fc' and isinstance(child, nn.modules.batchnorm._BatchNorm): child.eval()
            keep_frozen_bn_eval(child)
    model._keep_frozen_bn_eval = lambda: keep_frozen_bn_eval(model)
    return model


def embedding_resnet18():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()
    for p in model.parameters(): p.requires_grad = False
    return model.eval()


class SmallMLP(nn.Module):
    def __init__(self, input_dim=512, hidden=(256,64), num_classes=4, dropout=0.35):
        layers=[]; d=input_dim
        for h in hidden:
            layers += [nn.Linear(d,h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d=h
        layers.append(nn.Linear(d,num_classes))
        self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)
