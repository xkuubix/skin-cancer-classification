import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights


class DeepRadiomicsClassifier(nn.Module):
    def __init__(self, radiomic_feature_size=None, num_classes=None,
                 backbone='resnet34'):
        super(DeepRadiomicsClassifier, self).__init__()

        if backbone == 'efficientnet_b0':
            model = models.efficientnet_b0(weights='IMAGENET1K_V1')
            hidden_size = model.classifier[1].in_features
            model.classifier = nn.Identity()
        elif backbone == 'resnet34':
            model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            hidden_size = model.fc.in_features
            model.fc = nn.Identity()
        
        self.image_processor = model
        self.radiomic_norm = nn.BatchNorm1d(radiomic_feature_size)
        self.radiomic_fc = nn.Sequential(
            nn.Linear(radiomic_feature_size, hidden_size),
            nn.ReLU())
        
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, image, radiomic_features):
        image_features = self.image_processor(image)
        radiomic_features = self.radiomic_fc(self.radiomic_norm(radiomic_features))
        combined_features = torch.cat((image_features, radiomic_features), dim=1)
        output = self.classifier(self.dropout(combined_features))
        return output
    

class RadiomicsClassifier(nn.Module):
    def __init__(self, radiomic_feature_size=None, num_classes=None):
            super(RadiomicsClassifier, self).__init__()
            self.radiomic_norm = nn.BatchNorm1d(radiomic_feature_size)
            # self.radiomic_norm = nn.LayerNorm(radiomic_feature_size)
            self.radiomic_fc = nn.Sequential(
                nn.Linear(radiomic_feature_size, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU())
            self.dropout = nn.Dropout(0.5)
            self.classifier = nn.Linear(128, num_classes)

    def forward(self, radiomic_features):
        radiomic_features = self.radiomic_norm(radiomic_features)
        radiomic_features = self.radiomic_fc(radiomic_features)
        output = self.classifier(self.dropout(radiomic_features))
        return output
    
class ImageClassifier(nn.Module):
    def __init__(self, num_classes=None,
                 backbone='resnet34'):
        super(ImageClassifier, self).__init__()

        if backbone == 'efficientnet_b0':
            model = models.efficientnet_b0(weights='IMAGENET1K_V1')
            hidden_size = model.classifier[1].in_features
            model.classifier = nn.Linear(hidden_size, num_classes)
        elif backbone == 'resnet34':
            model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            hidden_size = model.fc.in_features
            model.fc = nn.Linear(hidden_size, num_classes)
        self.image_processor = model

    def forward(self, image):
        return self.image_processor(image)