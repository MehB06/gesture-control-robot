import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    CNN for image classification.

    Architecture Details:
    - 5 convolutional blocks with increasing number of channels (extract progressively more complex features).
    - Batch normalization and ReLU activation after each convolution (stabilizes training and introduces non-linearity).
    - Max pooling layers for spatial downsampling (reduces spatial dimensions to lower computation and increase receptive field).
    - Dropout for regularization in deeper blocks (prevents overfitting by randomly disabling neurons during training).
    - Global Average Pooling before the classifier (aggregates spatial information into a fixed-size vector).
    - Fully connected classifier with one hidden layer (maps extracted features to output class probabilities).
    """

    
    def __init__(self, in_channels=3, num_classes=29, base_channels=32, dropout=0.3):
        super().__init__()
        # Number of channels for each block (progressively doubling)
        c1, c2, c3, c4, c5 = base_channels, base_channels*2, base_channels*4, base_channels*8, base_channels*8
        
        # Block 1: 2 Conv layers + BatchNorm + ReLU + MaxPool (spatial downsampling) 
        # Block 2,3,4: similar to Block 1 but with Dropout for regularization and increased channels
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
        
        # Block 5: Final convolutional block without pooling but with dropout 
        self.block5 = nn.Sequential(
            nn.Conv2d(c4, c5, 3, padding=1, bias=False),
            nn.BatchNorm2d(c5),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Global Average Pooling: Converts feature map to vector by averaging spatial dimensions
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classifier: Fully connected layers with dropout for regularization
        self.classifier = nn.Sequential(
            nn.Linear(c5, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Custom weight initialization:
        - Convolutional layers use Kaiming normal initialization (good for ReLU activations).
        - BatchNorm layers are initialized to scale=1 and bias=0.
        """
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
        """
        Forward pass through the network:
        - Pass input through convolutional blocks sequentially.
        - Apply global average pooling.
        - Flatten the output and pass through classifier.
        - Return raw class logits.
        """
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
    """
    Helper function to create an instance of the CNN model with specified parameters.
    Args:
        in_channels (int): Number of input channels (3 for RGB images).
        num_classes (int): Number of output classes for classification.
        base_channels (int): Base number of channels for the first convolutional block.
        dropout (float): Dropout rate for regularization.
    """

    return CNN(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        dropout=dropout
    )