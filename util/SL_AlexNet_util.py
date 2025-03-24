# =============================================================================
# Utils --- AlexNet Client and Server Models
# ============================================================================

from torch import nn
import torch.nn.functional as F
import math

# =====================================================================================================
#                           AlexNet Client-side Model definition
# =====================================================================================================
class AlexNet_client_side(nn.Module):
    def __init__(self):
        super(AlexNet_client_side, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        resudial1 = F.relu(self.layer1(x))
        resudial2 = F.relu(resudial1)
        return resudial2


# =====================================================================================================
#                           AlexNet Server-side Model definition
# =====================================================================================================
class AlexNet_server_side(nn.Module):
    def __init__(self, dataset, num_classes):
        super(AlexNet_server_side, self).__init__()
        self.input_planes = 192

        self.layer4 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        if dataset == 'HAM' or 'MNIST':
            self.layer6 = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )

        if dataset == 'CIFAR10':
            self.layer6 = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=1, stride=1),
            )

        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256, 4096),  # Adjusted to match the output of the last conv layer
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x3 = F.relu(x)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)
        x7 = x6.view(x6.size(0), -1)
        y_hat = self.fc(x7)
        return y_hat