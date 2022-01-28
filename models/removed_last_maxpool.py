from models.bilinear_model import *
import torch.nn as nn
from models.stacked_attention_network import SANTimeDimMatrixAttOnModality1NLayer1Concat


def num_flat_features(x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class VggAcc79F174_7_RM(nn.Module):
    def __init__(self, in_channels, num_classes, fc_dim):
        """
        7 Layers CNN, early stage fusion with concatenation fusion method.
        @param in_channels: channels of input
        @param num_classes: 3 stage
        @param fc_dim: flattened representation dimension.
        """
        super(VggAcc79F174_7_RM, self).__init__()
        self.con_1 = nn.Conv1d(in_channels, 512, kernel_size=3, padding=1)
        self.con_2 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.con_3 = nn.Conv1d(512, 128, kernel_size=3, padding=1)
        self.con_4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)

        self.con_5 = nn.Conv1d(128, 512, kernel_size=3, padding=1)
        self.con_6 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.con_7 = nn.Conv1d(512, 512, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(fc_dim * 512, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.dense = nn.Linear(32, num_classes)

        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.25)

    def forward(self, x):
        output = self.con_1(x)
        output = self.relu(output)
        output = self.con_2(output)
        output = self.relu(output)
        output = self.max_pool(output)

        output = self.con_3(output)
        output = self.relu(output)
        output = self.con_4(output)
        output = self.relu(output)
        output = self.max_pool(output)

        output = self.con_5(output)
        output = self.relu(output)
        output = self.con_6(output)
        output = self.relu(output)
        output = self.con_7(output)
        output = self.relu(output)

        output = output.view(-1, num_flat_features(output))

        output = self.relu(self.fc1(output))
        output = self.drop_out(output)
        output = self.relu(self.fc2(output))
        output = self.drop_out(output)
        output = self.relu(self.fc3(output))
        output = self.drop_out(output)
        output = self.dense(output)
        return output


class VggAcc79F174NOFCRM(nn.Module):
    def __init__(self, in_channels):
        """
        this is the VGG backbone for hybrid fusion strategy, which removed all FC layers
        @param in_channels: the number of input channels. e.g., 7, 9
        """
        super(VggAcc79F174NOFCRM, self).__init__()
        self.con_1 = nn.Conv1d(in_channels, 512, kernel_size=3, padding=1)
        self.con_2 = nn.Conv1d(512, 512, kernel_size=3, padding=1)

        self.con_3 = nn.Conv1d(512, 128, kernel_size=3, padding=1)
        self.con_4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)

        self.con_5 = nn.Conv1d(128, 512, kernel_size=3, padding=1)
        self.con_6 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.con_7 = nn.Conv1d(512, 512, kernel_size=3, padding=1)

        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.25)

    def forward(self, x):
        output = self.con_1(x)
        output = self.relu(output)
        output = self.con_2(output)
        output = self.relu(output)
        output = self.max_pool(output)

        output = self.con_3(output)
        output = self.relu(output)
        output = self.con_4(output)
        output = self.relu(output)
        output = self.max_pool(output)

        output = self.con_5(output)
        output = self.relu(output)
        output = self.con_6(output)
        output = self.relu(output)
        output = self.con_7(output)
        output = self.relu(output)
        return output


class VggAcc79F174_RM_SplitModal(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, num_classes, fc_dim=25):
        """
        This is the hybrid fusion using the concatenation method
        @param in_channels_1: number of channels for activity counts should be 1
        @param in_channels_2:  number of channels for HRV features
        @param num_classes: 3 , three-stages
        @param fc_dim: the flattened feature map dim
        """
        super(VggAcc79F174_RM_SplitModal, self).__init__()
        self.stream_1 = VggAcc79F174NOFCRM(in_channels_1)  # act
        self.stream_2 = VggAcc79F174NOFCRM(in_channels_2)  # hrv
        self.fc1 = nn.Linear(fc_dim * 512 * 2, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.dense = nn.Linear(32, num_classes)
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.25)

    def forward(self, x):
        x_1 = x[:, 0, :].unsqueeze(dim=1)
        x_2 = x[:, 1:, :]
        output_1 = self.stream_1(x_1)
        output_2 = self.stream_2(x_2)
        concat_output = torch.cat((output_1, output_2), dim=1)
        feature = concat_output.view(-1, num_flat_features(concat_output))
        feature = self.relu(self.fc1(feature))
        feature = self.drop_out(feature)
        feature = self.relu(self.fc2(feature))
        feature = self.drop_out(feature)
        feature = self.relu(self.fc3(feature))
        feature = self.drop_out(feature)
        output = self.dense(feature)
        return feature, output


class VggAcc79F174_RM_SplitModalAdd(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, num_classes, fc_dim=25):
        """
        This is the hybrid fusion using the addition method
        @param in_channels_1: number of channels for activity counts should be 1
        @param in_channels_2:  number of channels for HRV features
        @param num_classes: 3 , three-stages
        @param fc_dim: the flattened feature map dim
        """
        super(VggAcc79F174_RM_SplitModalAdd, self).__init__()
        self.stream_1 = VggAcc79F174NOFCRM(in_channels_1)  # act
        self.stream_2 = VggAcc79F174NOFCRM(in_channels_2)  # hrv
        self.fc1 = nn.Linear(fc_dim * 512, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.dense = nn.Linear(32, num_classes)
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.25)

    def forward(self, x):
        x_1 = x[:, 0, :].unsqueeze(dim=1)
        x_2 = x[:, 1:, :]
        output_1 = self.stream_1(x_1)
        output_2 = self.stream_2(x_2)

        concat_output = output_1+output_2
        feature = concat_output.view(-1, num_flat_features(concat_output))
        feature = self.relu(self.fc1(feature))
        feature = self.drop_out(feature)
        feature = self.relu(self.fc2(feature))
        feature = self.drop_out(feature)
        feature = self.relu(self.fc3(feature))
        feature = self.drop_out(feature)
        output = self.dense(feature)
        return feature, output


class VggAcc79F174_RM_SANTiDimMatAttMod1NLayer1Con(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, num_classes, att_on_modality, time_step_dim=5):
        """
        This is the hybrid fusion using the attention method
        @param in_channels_1: number of channels for activity counts should be
        @param in_channels_2: number of channels for HRV features
        @param num_classes: 3, three-stages
        @param att_on_modality: which modality we should to pay attention
        @param time_step_dim: the flattened feature map dim
        """
        super(VggAcc79F174_RM_SANTiDimMatAttMod1NLayer1Con, self).__init__()
        self.stream_1 = VggAcc79F174NOFCRM(in_channels_1)  # act
        self.stream_2 = VggAcc79F174NOFCRM(in_channels_2)  # hrv
        self.san = SANTimeDimMatrixAttOnModality1NLayer1Concat(input_feature_dim=512, attention_dim=256,
                                                               time_step_dim=time_step_dim)
        self.fc1 = nn.Linear(time_step_dim * 512 * 2, 512)  # was 12 x 512 * 2 now is 12 x 512
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.dense = nn.Linear(32, num_classes)
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.25)
        assert att_on_modality in ["act", "car"]
        self.att_on_modality = att_on_modality

    def forward(self, x):
        x_1 = x[:, 0, :].unsqueeze(dim=1)
        x_2 = x[:, 1:, :]
        output_1 = self.stream_1(x_1)
        output_2 = self.stream_2(x_2)
        if self.att_on_modality == "act":
            att_output = self.san(output_1, output_2)
        else:
            att_output = self.san(output_2, output_1)
        feature = att_output.view(-1, num_flat_features(att_output))
        feature = self.drop_out(self.relu(self.fc1(feature)))
        feature = self.relu(self.fc2(feature))
        feature = self.drop_out(feature)
        feature = self.relu(self.fc3(feature))
        feature = self.drop_out(feature)
        output = self.dense(feature)
        return feature, output


class VggAcc79F174_RM_BiLinear(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, num_classes):
        """
        This is the hybrid fusion using the bilinear fusion method
        @param in_channels_1: number of channels for activity counts should be
        @param in_channels_2: number of channels for HRV features
        @param num_classes: 3, three-stages
        """
        super(VggAcc79F174_RM_BiLinear, self).__init__()
        self.stream_1 = VggAcc79F174NOFCRM(in_channels_1)  # act
        self.stream_2 = VggAcc79F174NOFCRM(in_channels_2)  # hrv
        self.bl = BilinearFusion(512, 1024)
        self.fc1 = nn.Linear(2 * 512, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.dense = nn.Linear(32, num_classes)
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.25)

    def forward(self, x):
        x_1 = x[:, 0, :].unsqueeze(dim=1)
        x_2 = x[:, 1:, :]
        output_1 = self.stream_1(x_1)
        output_2 = self.stream_2(x_2)
        att_output = self.bl(output_1, output_2)
        feature = self.drop_out(self.relu(self.fc1(att_output)))
        feature = self.relu(self.fc2(feature))
        feature = self.drop_out(feature)
        feature = self.relu(self.fc3(feature))
        feature = self.drop_out(feature)
        output = self.dense(feature)
        return feature, output


