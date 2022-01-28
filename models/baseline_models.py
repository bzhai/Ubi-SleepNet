import torch.nn as nn
import torch

def num_flat_features(x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class CNN(nn.Module):
    def __init__(self, in_channels, num_classes, init_weights: bool = True):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=2),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(6400, num_classes),
            # nn.ReLU(),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
