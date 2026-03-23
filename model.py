import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_pretrained_model(num_classes=1000):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    return model, weights.transforms()
