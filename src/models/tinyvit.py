import timm
import torch.nn as nn
import torchvision.models as models
class MultiOutputTinyVit(nn.Module):
    def __init__(self, num_classes):
        super(MultiOutputTinyVit, self).__init__()
        base_model_name = 'tiny_vit_11m_224'
        # Remove default head
        self.backbone = timm.create_model(base_model_name, pretrained=True, num_classes=0)  
        self.dropout = nn.Dropout(0.15)
        self.fc = nn.Linear(448, num_classes)
    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        pred = self.fc(features)
        return pred