import torch.nn as nn
import torchvision.models as models

class MultiOutputRegNet(nn.Module):
    def __init__(self, num_classes):
        super(MultiOutputRegNet, self).__init__()
        self.backbone = models.regnet_x_400mf(weights=models.RegNet_X_400MF_Weights.IMAGENET1K_V1)
        # Remove the final classification head
        self.backbone.fc = nn.Identity()  
        self.dropout = nn.Dropout(0.15)
        self.fc = nn.Linear(400, num_classes)  # 1280 is the output size from the feature extractor

    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        pred = self.fc(features)
        return pred
