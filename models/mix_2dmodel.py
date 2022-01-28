import torch.nn as nn
import torch


def num_flat_features(x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class VggIMG(nn.Module):
    """
    Treat inputs as a image using 3x1 kernel, e.g. mesa H=9, W=101
    """
    def __init__(self, in_channels=9, num_classes=3, time_step_dim=25) -> None:
        super(VggIMG, self).__init__()
        self.channels = in_channels
        self.features = nn.Sequential(
            # #### block 1
            nn.Conv2d(1, 512, kernel_size=(1, 3), padding=(0, 1)),  # con_1
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(1, 3), padding=(0, 1)),  # con_2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # max_pool
            # #### block 2
            nn.Conv2d(512, 128, kernel_size=(1, 3), padding=(0, 1)),  # con_3
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(1, 3), padding=(0, 1)),  # con_4
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # max pool
            # #### block 3 = con_3
            nn.Conv2d(128, 512, kernel_size=(1, 3), padding=(0, 1)),  # con_5
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(1, 3), padding=(0, 1)),  # con_6
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(1, 3), padding=(0, 1)),  # con_7
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(time_step_dim * 512 * self.channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class VggIMGRes(nn.Module):
    """
    Treat inputs as a image using 3x1 kernel, e.g. mesa H=9, W=101
    """
    def __init__(self, in_channels=9, num_classes=3, time_step_dim=25) -> None:
        super(VggIMGRes, self).__init__()
        self.conv_1 = nn.Conv2d(1, 512, kernel_size=(1, 3), padding=(0, 1))  # con_1
        self.conv_2 = nn.Conv2d(512, 512, kernel_size=(1, 3), padding=(0, 1))  # con_2
        self.conv_3 = nn.Conv2d(512, 128, kernel_size=(1, 3), padding=(0, 1))  # conv_3
        self.conv_4 = nn.Conv2d(128, 128, kernel_size=(1, 3), padding=(0, 1))  # conv_4
        self.conv_5 = nn.Conv2d(128, 512, kernel_size=(1, 3), padding=(0, 1))  # conv_5
        self.conv_6 = nn.Conv2d(512, 512, kernel_size=(1, 3), padding=(0, 1))  # conv_6
        self.conv_7 = nn.Conv2d(512, 512, kernel_size=(1, 3), padding=(0, 1))  # conv_7
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.channels = in_channels

        self.classifier = nn.Sequential(
            nn.Linear(time_step_dim * 512 * self.channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        x = self.relu(x)
        residual_1 = x
        output = self.conv_2(x)
        output += residual_1
        output = self.relu(output)
        output = self.max_pool(output)

        output_2 = self.conv_3(output)
        residual_2 = output_2
        output_2 = self.relu(output_2)
        output_2 = self.conv_4(output_2)
        output_2 += residual_2
        output_2 = self.relu(output_2)
        output_2 = self.max_pool(output_2)

        output_3 = self.conv_5(output_2)
        output_3 = self.relu(output_3)
        residual_3 = output_3
        output_3 = self.conv_6(output_3)
        output_3 = self.relu(output_3)
        output_3 = self.conv_7(output_3)
        output_3 += residual_3
        output_3 = self.relu(output_3)

        output_3 = torch.flatten(output_3, 1)
        output_3 = self.classifier(output_3)
        return output_3


class VggIMGResSum(nn.Module):
    """
    Treat inputs as a image using 3x1 kernel, e.g. mesa H=9, W=101
    """
    def __init__(self, in_channels=9, num_classes=3, time_step_dim=25) -> None:
        super(VggIMGResSum, self).__init__()
        self.conv_1 = nn.Conv2d(1, 512, kernel_size=(1, 3), padding=(0, 1))  # con_1
        self.conv_2 = nn.Conv2d(512, 512, kernel_size=(1, 3), padding=(0, 1))  # con_2
        self.conv_3 = nn.Conv2d(512, 128, kernel_size=(1, 3), padding=(0, 1))  # conv_3
        self.conv_4 = nn.Conv2d(128, 128, kernel_size=(1, 3), padding=(0, 1))  # conv_4
        self.conv_5 = nn.Conv2d(128, 512, kernel_size=(1, 3), padding=(0, 1))  # conv_5
        self.conv_6 = nn.Conv2d(512, 512, kernel_size=(1, 3), padding=(0, 1))  # conv_6
        self.conv_7 = nn.Conv2d(512, 512, kernel_size=(1, 3), padding=(0, 1))  # conv_7
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.channels = in_channels

        self.classifier = nn.Sequential(
            nn.Linear(time_step_dim * 512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        x = self.relu(x)
        residual_1 = x
        output = self.conv_2(x)
        output += residual_1
        output = self.relu(output)
        output = self.max_pool(output)

        output_2 = self.conv_3(output)
        residual_2 = output_2
        output_2 = self.relu(output_2)
        output_2 = self.conv_4(output_2)
        output_2 += residual_2
        output_2 = self.relu(output_2)
        output_2 = self.max_pool(output_2)

        output_3 = self.conv_5(output_2)
        output_3 = self.relu(output_3)
        residual_3 = output_3
        output_3 = self.conv_6(output_3)
        output_3 = self.relu(output_3)
        output_3 = self.conv_7(output_3)
        output_3 += residual_3
        output_3 = self.relu(output_3)
        output_3 = output_3.sum(axis=2)
        output_3 = torch.flatten(output_3, 1)
        output_3 = self.classifier(output_3)
        return output_3


class VggIMGSum(nn.Module):
    """
    Treat inputs as a image using 3x1 kernel, e.g. mesa H=9, W=101
    """
    def __init__(self, in_channels=9, num_classes=3, time_step_dim=25) -> None:
        super(VggIMGSum, self).__init__()
        self.channels = in_channels
        self.features = nn.Sequential(
            # #### block 1
            nn.Conv2d(1, 512, kernel_size=(1, 3), padding=(0, 1)),  # con_1
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(1, 3), padding=(0, 1)),  # con_2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # max_pool
            # #### block 2
            nn.Conv2d(512, 128, kernel_size=(1, 3), padding=(0, 1)),  # con_3
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(1, 3), padding=(0, 1)),  # con_4
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # max pool
            # #### block 3 = con_3
            nn.Conv2d(128, 512, kernel_size=(1, 3), padding=(0, 1)),  # con_5
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(1, 3), padding=(0, 1)),  # con_6
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(1, 3), padding=(0, 1)),  # con_7
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(time_step_dim * 512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.sum(axis=2)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class VggIMGNOFC(nn.Module):
    """
    Treat inputs as a image using 3x1 kernel, e.g. mesa H=9, W=101
    """
    def __init__(self) -> None:
        super(VggIMGNOFC, self).__init__()
        self.features = nn.Sequential(
            # #### block 1
            nn.Conv2d(1, 512, kernel_size=(1, 3), padding=(0, 1)),  # con_1
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(1, 3), padding=(0, 1)),  # con_2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # max_pool
            # #### block 2
            nn.Conv2d(512, 128, kernel_size=(1, 3), padding=(0, 1)),  # con_3
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(1, 3), padding=(0, 1)),  # con_4
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # max pool
            # #### block 3 = con_3
            nn.Conv2d(128, 512, kernel_size=(1, 3), padding=(0, 1)),  # con_5
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(1, 3), padding=(0, 1)),  # con_6
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(1, 3), padding=(0, 1)),  # con_7
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return x


class VggIMGResNOFC(nn.Module):
    """
    Treat inputs as a image using 3x1 kernel, e.g. mesa H=9, W=101
    """
    def __init__(self, in_channels=9, num_classes=3, time_step_dim=25) -> None:
        super(VggIMGResNOFC, self).__init__()
        self.conv_1 = nn.Conv2d(1, 512, kernel_size=(1, 3), padding=(0, 1))  # con_1
        self.conv_2 = nn.Conv2d(512, 512, kernel_size=(1, 3), padding=(0, 1))  # con_2
        self.conv_3 = nn.Conv2d(512, 128, kernel_size=(1, 3), padding=(0, 1))  # conv_3
        self.conv_4 = nn.Conv2d(128, 128, kernel_size=(1, 3), padding=(0, 1))  # conv_4
        self.conv_5 = nn.Conv2d(128, 512, kernel_size=(1, 3), padding=(0, 1))  # conv_5
        self.conv_6 = nn.Conv2d(512, 512, kernel_size=(1, 3), padding=(0, 1))  # conv_6
        self.conv_7 = nn.Conv2d(512, 512, kernel_size=(1, 3), padding=(0, 1))  # conv_7
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        x = self.relu(x)
        residual_1 = x
        output = self.conv_2(x)
        output += residual_1
        output = self.relu(output)
        output = self.max_pool(output)

        output_2 = self.conv_3(output)
        residual_2 = output_2
        output_2 = self.relu(output_2)
        output_2 = self.conv_4(output_2)
        output_2 += residual_2
        output_2 = self.relu(output_2)
        output_2 = self.max_pool(output_2)

        output_3 = self.conv_5(output_2)
        output_3 = self.relu(output_3)
        residual_3 = output_3
        output_3 = self.conv_6(output_3)
        output_3 = self.relu(output_3)
        output_3 = self.conv_7(output_3)
        output_3 += residual_3
        output_3 = self.relu(output_3)
        return output_3
