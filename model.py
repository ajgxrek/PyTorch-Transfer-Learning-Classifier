import torch.nn as nn
from torchvision import models


def get_model():
    # Use pre-trained weights from ImageNet
    model = models.resnet18(weights='DEFAULT')

    # Freeze all layers to use them as a fixed feature extractor
    for param in model.parameters():
        param.requires_grad = False

    # Replace the original 1000-class head with a binary classifier
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    return model