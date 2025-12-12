from torchvision import models
import torch.nn as nn

def build_model(num_classes, pretrained=False):
    model = models.resnet18(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model