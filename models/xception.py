import torch.nn as nn
import timm

def get_xception(num_classes=7):
    model = timm.create_model("xception", pretrained=True)
    in_features = model.get_classifier().in_features
    model.fc = nn.Linear(in_features, num_classes)  # replace final layer
    return model
