import torch
import torch.nn as nn
from torchvision.models import resnet18

def get_model():
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model
