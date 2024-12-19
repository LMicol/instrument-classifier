# I forgot to do .state_dict() when saving
# This script is just to fix this problem to use the fifth model properly without the security risk
import torch
import torch.nn as  nn
import torch.nn.functional as F

N_MFCC = 15
DURATION = 2
TARGET_SR = 44100

class AudioClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AudioClassifier, self).__init__()

        # CNN layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) 
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Halves the dimensions

        # Fully connected layers
        self.fc1 = nn.Linear(64 * (N_MFCC // 8) * (DURATION * TARGET_SR // 8 // 512), 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten the tensor
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers

        # Fully connected layers with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

model = torch.load('5_model/fifth_idea.pth', map_location=torch.device('cpu'))
model.eval()
torch.save(model.state_dict(), 'model.pth')