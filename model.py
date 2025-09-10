import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a CNN by subclassing nn.Module
class SimpleCNN(nn.Module): 
    def __init__(self, num_classes=10):
        # Call the parent class (nn.Module)
        super(SimpleCNN, self).__init__()

        # First convolutional layer:
        # input has 3 channels (RGB), we create 32 feature maps
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)

        # Second convolutional layer: 
        # takes the 32 feature maps from conv1 and produces 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Max pooling layer: reduce width/height by a factor of 2
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected (dense) layer:
        # input size = 64 feature maps of size 8x8
        self.fc1 = nn.Linear(64 * 8 * 8, 128)

        # Ouput layer: 
        # 128 features -> number of classes (CIFAR-10 has 10 classes)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Pass through conv1 -> ReLU -> pooling
        x = self.pool(F.relu(self.conv1(x)))

        # Pass through conv2 -> ReLU -> pooling
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten tensor into a vector: [batch, 64*8*8]
        x = x.view(-1, 64 * 8 * 8)

        # First dense layer + ReLU
        x = F.relu(self1.fc1(x))

        # Ouput layer (no softmax, CrossEntropyLoss will handle that)
        x = self.fc2(x)

        return x