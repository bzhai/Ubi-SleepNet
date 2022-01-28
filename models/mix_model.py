import torch.nn as nn
from models.bilinear_model import *
from torchsummary import summary

from models.stacked_attention_network import SANTimeDimMatrixAttOnModality1NLayer1Concat
from utilities.utils import num_flat_features


class VggAcc79F174ResdPlus(nn.Module):
    def __init__(self, in_channels, num_classes, time_step_dim):
        """
        This is the ResNet model in early stage fusion using the concatenation method.
        @param in_channels: the total number of input channels
        @param num_classes: 3 ,three stage sleep classification
        @param time_step_dim: the flattened feature map dim
        """
        super(VggAcc79F174ResdPlus, self).__init__()
        self.con_1 = nn.Conv1d(in_channels, 512, kernel_size=3, padding=1)
        # self.bn_1 = nn.BatchNorm1d(512)
        self.con_2 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        # self.bn_2 = nn.BatchNorm1d(512)

        self.con_3 = nn.Conv1d(512, 128, kernel_size=3, padding=1)
        # self.bn_3 = nn.BatchNorm1d(128)
        self.con_4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        # self.bn_4 = nn.BatchNorm1d(128)

        self.con_5 = nn.Conv1d(128, 512, kernel_size=3, padding=1)
        # self.bn_5 = nn.BatchNorm1d(512)
        self.con_6 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        # self.bn_6 = nn.BatchNorm1d(512)
        self.con_7 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        # self.bn_7 = nn.BatchNorm1d(512)

        self.fc1 = nn.Linear(time_step_dim * 512, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.dense = nn.Linear(32, num_classes)

        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.con_1(x)
        x = self.relu(x)
        residual_1 = x
        output = self.con_2(x)
        output += residual_1
        output = self.relu(output)
        output = self.max_pool(output)
        output_2 = self.con_3(output)
        residual_2 = output_2

        output_2 = self.relu(output_2)
        output_2 = self.con_4(output_2)
        output_2 += residual_2
        output_2 = self.relu(output_2)
        output_2 = self.max_pool(output_2)
        output_3 = self.con_5(output_2)
        output_3 = self.relu(output_3)
        residual_3 = output_3
        output_3 = self.con_6(output_3)
        output_3 = self.relu(output_3)

        output_3 = self.con_7(output_3)
        output_3 += residual_3
        output_3 = self.relu(output_3)
        output_3 = output_3.view(-1, num_flat_features(output_3))

        output_3 = self.relu(self.fc1(output_3))
        output_3 = self.drop_out(output_3)
        output_3 = self.relu(self.fc2(output_3))
        feature = self.drop_out(self.relu(self.fc3(output_3)))
        dense_output = self.dense(feature)

        return feature, dense_output


class ResPlusSingleModalityNOFC(nn.Module):
    def __init__(self, in_channels):
        """
        This is the ResNet backbone network without fully connected layers for early stage fusion and hybrid fusion
        @param in_channels: number of input channels
        """
        super(ResPlusSingleModalityNOFC, self).__init__()
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
        x = self.con_1(x)
        x = self.relu(x)
        residual_1 = x
        output = self.con_2(x)
        output += residual_1
        output = self.relu(output)
        output = self.max_pool(output)
        output_2 = self.con_3(output)
        output_2 = self.relu(output_2)
        residual_2 = output_2
        output_2 = self.con_4(output_2)
        output_2 += residual_2
        output_2 = self.relu(output_2)
        output_2 = self.max_pool(output_2)
        output_3 = self.con_5(output_2)
        output_3 = self.relu(output_3)
        residual_3 = output_3
        output_3 = self.con_6(output_3)
        output_3 = self.relu(output_3)

        output_3 = self.con_7(output_3)
        output_3 += residual_3
        output_3 = self.relu(output_3)

        return output_3


class ResPlusSplitModalCon(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, num_classes, time_step_dim=25):
        """
        This is the model in hybrid fusion strategy using the concatenation method
        @param in_channels_1: the total number of input channels for the activity counts
        @param in_channels_2: the total number of input channels for HRV features
        @param num_classes: 3 ,three stage sleep classification
        @param time_step_dim: the flattened feature map dim
        """
        super(ResPlusSplitModalCon, self).__init__()
        self.stream_1 = ResPlusSingleModalityNOFC(in_channels_1)  # act
        self.stream_2 = ResPlusSingleModalityNOFC(in_channels_2)  # hrv
        self.fc1 = nn.Linear(time_step_dim * 512 * 2, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.dense = nn.Linear(32, num_classes)
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.25)

    def forward(self, x):
        x_1 = x[:, 0, :].unsqueeze(dim=1)  # act modality
        x_2 = x[:, 1:, :]  # hrv modality
        output_1 = self.stream_1(x_1)
        output_2 = self.stream_2(x_2)
        concat_output = torch.cat((output_1, output_2), dim=1)
        feature = concat_output.view(-1, num_flat_features(concat_output))
        feature = self.drop_out(self.relu(self.fc1(feature)))
        feature = self.relu(self.fc2(feature))
        feature = self.drop_out(feature)
        feature = self.relu(self.fc3(feature))
        feature = self.drop_out(feature)
        output = self.dense(feature)
        return feature, output


class ResPlusSplitModalPlus(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, num_classes, time_step_dim=25):
        """
        This is the model in hybrid fusion strategy using the addition method
        @param in_channels_1: the total number of input channels for the activity counts
        @param in_channels_2: the total number of input channels for HRV features
        @param num_classes: 3 ,three stage sleep classification
        @param time_step_dim: the flattened feature map dim
        """
        super(ResPlusSplitModalPlus, self).__init__()
        self.stream_1 = ResPlusSingleModalityNOFC(in_channels_1)  # act
        self.stream_2 = ResPlusSingleModalityNOFC(in_channels_2)  # hrv
        self.fc1 = nn.Linear(time_step_dim * 512, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.dense = nn.Linear(32, num_classes)
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.25)

    def forward(self, x):
        x_1 = x[:, 0, :].unsqueeze(dim=1)  # act modality
        x_2 = x[:, 1:, :]  # hrv modality
        output_1 = self.stream_1(x_1)
        output_2 = self.stream_2(x_2)

        concat_output = output_1 + output_2
        feature = concat_output.view(-1, num_flat_features(concat_output))
        feature = self.drop_out(self.relu(self.fc1(feature)))
        feature = self.relu(self.fc2(feature))
        feature = self.drop_out(feature)
        feature = self.relu(self.fc3(feature))
        feature = self.drop_out(feature)
        output = self.dense(feature)
        return feature, output


class ResPlusSplitModal_BiLinear(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, num_classes):
        """
        This is the model in hybrid fusion strategy using the bilinear method
        @param in_channels_1: the total number of input channels for the activity counts
        @param in_channels_2: the total number of input channels for HRV features
        @param num_classes: 3 ,three stage sleep classification
        """
        super(ResPlusSplitModal_BiLinear, self).__init__()
        self.stream_1 = ResPlusSingleModalityNOFC(in_channels_1)
        self.stream_2 = ResPlusSingleModalityNOFC(in_channels_2)
        self.bl = BilinearFusion(512, 1024)
        self.fc1 = nn.Linear(1024, 512)  # bilinear output is 1024
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.dense = nn.Linear(32, num_classes)
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.25)

    def forward(self, x):
        x_1 = x[:, 0, :].unsqueeze(dim=1)  # act modality
        x_2 = x[:, 1:, :]  # hrv modality
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


class ResPlusSplitModal_SANTiDimMatAttMod1NLayer1Con(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, num_classes, att_on_modality, time_step_dim=25):
        """
        This is the model in hybrid fusion strategy using the concatenation method
        @param in_channels_1: the total number of input channels for the activity counts
        @param in_channels_2: the total number of input channels for HRV features
        @param num_classes: 3 ,three stage sleep classification
        @param time_step_dim: the flattened feature map dim
        @param att_on_modality: the modality to apply attention method
        """
        super(ResPlusSplitModal_SANTiDimMatAttMod1NLayer1Con, self).__init__()
        self.stream_1 = ResPlusSingleModalityNOFC(in_channels_1)  # act
        self.stream_2 = ResPlusSingleModalityNOFC(in_channels_2)  # hrv
        self.san = SANTimeDimMatrixAttOnModality1NLayer1Concat(input_feature_dim=512, attention_dim=256,
                                                               time_step_dim=time_step_dim)
        self.fc1 = nn.Linear(time_step_dim * 512 * 2, 512)  # the concatenation will double 512
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.dense = nn.Linear(32, num_classes)
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.25)
        assert att_on_modality in ["act", "car"]
        self.att_on_modality = att_on_modality

    def forward(self, x):
        x_1 = x[:, 0, :].unsqueeze(dim=1)  # act modality
        x_2 = x[:, 1:, :]  # hrv modality
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


# if __name__ == "__main__":

    # print("output shape %s", output1.shape)

    # model1 = VggAcc79F174_SplitModal_SANTimeDimMatrixAttOnMod1NLayer1(in_channels_1=1, in_channels_2=8, num_classes=3)
    # model1 = ResConcatSplitModal_SANTimeDimMatrixAttOnMod1NLayer1(in_channels_1=1, in_channels_2=8, num_classes=3,
    #                                                               att_on_modality='act')
    # model1 = ResConcatSplitModal_BiLinear(in_channels_1=1, in_channels_2=8, num_classes=3)
    # model1 = ResConcatSplitModal_SANTimeDimMatrixAttOnMod1NLayer1Concat(in_channels_1=1, in_channels_2=8, num_classes=3, att_on_modality="act")
    # input1 = torch.rand(2, 7, 21)
    # input_dim = 7
    # input_dim1 = 1
    # input_dim2 = input_dim - 1
    # model1 = ResPlusSplitModalPlus(input_dim1, input_dim2, 3)
    # output1 = model1(input1)
    # summary(model1, (input_dim, 101),  device="cpu")
    #
    # model1 = ResPlusSplitModalEleMul(input_dim1, input_dim2, 3)
    # summary(model1, (input_dim, 101),  device="cpu")
    #
    # model1 = ResPlusSplitModalEleMul(input_dim1, input_dim2, 3)
    # summary(model1, (input_dim, 101),  device="cpu")
    #
    # model1 = ResPlusSplitModalCon(input_dim1, input_dim2, 3)
    # summary(model1, (input_dim, 101),  device="cpu")

    # model1 = ResPlusSplitModal_SANTiDimMatAttMod1NLayer1Con(input_dim1, input_dim2, 3, att_on_modality='act')
    # summary(model1, (input_dim, 101),  device="cpu")
    #
    # model1 = ResPlusSplitModal_BiLinear(input_dim1, input_dim2, 3)
    # summary(model1, (input_dim, 101),  device="cpu")

    # output1 = model1(input1)
    # output1 = model1(input1[:, 0, :].unsqueeze(dim=1) )
    # print("output shape %s", output1[1].shape)
