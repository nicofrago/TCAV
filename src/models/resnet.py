import torch.nn as nn
import torchvision.models as models
class MultiOutputResNet(nn.Module):
    def __init__(self, num_classes):
        super(MultiOutputResNet, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()  # Remove final FC layer
        
        self.dropout = nn.Dropout(0.15)
        self.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        pred = self.fc(features)
        return pred