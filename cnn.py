import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=29, base_channels=32, dropout=0.3):
        super().__init__()
        c1, c2, c3, c4, c5 = base_channels, base_channels*2, base_channels*4, base_channels*8, base_channels*8
        
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, 3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout)
        )
        
        self.block4 = nn.Sequential(
            nn.Conv2d(c3, c4, 3, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
            nn.Conv2d(c4, c4, 3, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout)
        )
        
        self.block5 = nn.Sequential(
            nn.Conv2d(c4, c5, 3, padding=1, bias=False),
            nn.BatchNorm2d(c5),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(c5, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_model(in_channels=3, num_classes=29, base_channels=32, dropout=0.3):
    return CNN(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        dropout=dropout
    )