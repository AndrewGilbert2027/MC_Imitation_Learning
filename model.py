import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetLikeModel(nn.Module):
    def __init__(self, num_classes=11):  # Adjusted num_classes to 11 for binary camera directions
        super(ResNetLikeModel, self).__init__()
        # Initial convolution layer
        self.conv1 = nn.Conv2d(60, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Updated input channels to 60 (20 frames x 3 RGB channels)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual block 1
        self.layer1 = self._make_layer(64, 64, 3)  # Increased number of blocks to 3
        # Residual block 2
        self.layer2 = self._make_layer(64, 128, 4, stride=2)  # Increased number of blocks to 4
        # Residual block 3
        self.layer3 = self._make_layer(128, 256, 4, stride=2)  # Increased number of blocks to 4
        # Residual block 4
        self.layer4 = self._make_layer(256, 512, 5, stride=2)  # Increased number of blocks to 5
        # Residual block 5 (new layer added)
        self.layer5 = self._make_layer(512, 1024, 2, stride=2)  # Added new residual block

        # Global average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.1)  # Reduced dropout rate to 10%
        self.fc = nn.Linear(1024, num_classes)  # Updated input size to match new layer

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)  # Forward pass through the new layer

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)  # Apply dropout
        x = self.fc(x)
        return x
    
    def predict(self, x):
        """
        Predicts the output for the input tensor x.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 60, 124, 124).
        
        Returns:
            torch.Tensor: Output predictions of shape (batch_size, num_classes).
        """
        self.eval()
        with torch.no_grad():
            x = self.forward(x)
            return x
        


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out

# Example usage
if __name__ == "__main__":
    model = ResNetLikeModel(num_classes=11)  # Adjust num_classes based on your feature vector size
    print(model)