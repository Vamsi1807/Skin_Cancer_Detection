from torchvision import models
import torch.nn as nn

def get_resnet50(num_classes=7):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # ✅ new API
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
