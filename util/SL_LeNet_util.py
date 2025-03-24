# =============================================================================
# Utils --- LeNet Client and Server Models
# ============================================================================

from torch import nn
import torch.nn.functional as F
import math

# =====================================================================================================
#                           LeNet Client-side Model definition
# =====================================================================================================
class LeNet_client_side(nn.Module):
    def __init__(self):
        super(LeNet_client_side, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1),
            nn.Conv2d(3, 16, kernel_size=5)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        resudial2 = F.relu(self.layer1(x))
        return resudial2


# =====================================================================================================
#                           LeNet Server-side Model definition
# =====================================================================================================
class LeNet_server_side(nn.Module):
    def __init__(self, num_classes, dataset):
        super(LeNet_server_side, self).__init__()
        if dataset == 'HAM':
            self.fc1 = nn.Linear(16 * 14 * 14, 120)
        if dataset == 'MNIST':
            self.fc1 = nn.Linear(16 * 30 * 30, 120)
        if dataset == 'CIFAR10':
            self.fc1 = nn.Linear(16 * 14 * 14, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y_hat = self.fc3(x)
        return y_hat