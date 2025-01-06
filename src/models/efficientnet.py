import torch.nn as nn
import torchvision.models as models
class MultiOutputEfficientNetB0(nn.Module):
    def __init__(self, num_classes):
        super(MultiOutputEfficientNetB0, self).__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.backbone.classifier = nn.Identity()  # Remove final layer
        self.dropout = nn.Dropout(0.15)
        self.fc = nn.Linear(1280, num_classes)
    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        pred = self.fc(features)
        return pred
    
class MultiOutputEfficientNetB1(nn.Module):
    def __init__(self, num_classes):
        super(MultiOutputEfficientNetB1, self).__init__()
        self.backbone = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
        self.backbone.classifier = nn.Identity()  # Remove final layer
        self.dropout = nn.Dropout(0.15)
        self.fc= nn.Linear(1280, num_classes)
    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        pred = self.fc(features)
        return pred

class MultiOutputEfficientNetB2(nn.Module):
    def __init__(self, num_classes):
        super(MultiOutputEfficientNetB2, self).__init__()
        self.backbone = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        self.backbone.classifier = nn.Identity()  # Remove final layer
        self.dropout = nn.Dropout(0.15)
        self.fc = nn.Linear(1408, num_classes)
    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        pred = self.fc(features)
        return pred