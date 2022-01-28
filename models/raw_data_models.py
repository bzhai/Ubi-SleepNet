import torch.nn as nn
import torch

from models.bilinear_model import BilinearFusion
from models.mix_model import ResPlusSingleModalityNOFC
from models.removed_last_maxpool import VggAcc79F174NOFCRM
from models.stacked_attention_network import SANTimeDimMatrixAttOnModality1NLayer1Concat
from utilities.utils import num_flat_features
from models.mix_2dmodel import VggIMGNOFC, VggIMGResNOFC


class VggAcc79F174_7_RM_Raw_Appl_1(nn.Module):
    def __init__(self, in_channels, raw_acc_in_ch, num_classes, fc_dim):
        """
        this is the Vgg model for the early stage fusion strategy, which use the raw accelerometer data. We extract the
        representation from the raw accelerometer then concatenate it with the HRV features for feature learning.
        @param in_channels: the input channel
        @param raw_acc_in_ch: 3, three axis as the input channel
        @param num_classes: 3, three-stage sleep classification
        @param fc_dim: the flattened representation dimension.
        """
        super(VggAcc79F174_7_RM_Raw_Appl_1, self).__init__()

        self.con_1 = nn.Conv1d(in_channels, 512, kernel_size=3, padding=1)
        self.con_2 = nn.Conv1d(512, 512, kernel_size=3, padding=1)

        self.con_3 = nn.Conv1d(512, 128, kernel_size=3, padding=1)
        self.con_4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)

        self.con_5 = nn.Conv1d(128, 512, kernel_size=3, padding=1)
        self.con_6 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.con_7 = nn.Conv1d(512, 512, kernel_size=3, padding=1)

        self.raw_conv_1 = nn.Conv1d(raw_acc_in_ch, 512, kernel_size=3, stride=3)
        self.raw_conv_2 = nn.Conv1d(512, 512, kernel_size=3, stride=3)
        self.raw_conv_3 = nn.Conv1d(512, 512, kernel_size=3, stride=3)

        self.raw_conv_4 = nn.Conv1d(512, 128, kernel_size=12, stride=1)
        self.raw_conv_5 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.raw_conv_6 = nn.Conv1d(128, 1, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(fc_dim * 512, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.dense = nn.Linear(32, num_classes)

        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.cls_relu = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.25)
        self.raw_drop_out = nn.Dropout()

    def forward(self, acc_x, hrv_x):
        """
        @param acc_x: raw accelerometer data
        @param hrv_x: hrv features
        @return: logits and features
        """
        acc_output = self.relu(self.raw_conv_1(acc_x))
        acc_output = self.relu(self.raw_conv_2(acc_output))
        acc_output = self.relu(self.raw_conv_3(acc_output))
        acc_output = self.raw_drop_out(acc_output)
        acc_output = self.relu(self.raw_conv_4(acc_output))
        acc_output = self.relu(self.raw_conv_5(acc_output))
        acc_output = self.relu(self.raw_conv_6(acc_output))
        acc_output = self.raw_drop_out(acc_output)
        hrv_output = torch.cat((acc_output, hrv_x), dim=1)
        hrv_output = self.con_1(hrv_output)
        hrv_output = self.relu(hrv_output)
        hrv_output = self.con_2(hrv_output)
        hrv_output = self.relu(hrv_output)
        hrv_output = self.max_pool(hrv_output)

        hrv_output = self.con_3(hrv_output)
        hrv_output = self.relu(hrv_output)
        hrv_output = self.con_4(hrv_output)
        hrv_output = self.relu(hrv_output)
        hrv_output = self.max_pool(hrv_output)

        hrv_output = self.con_5(hrv_output)
        hrv_output = self.relu(hrv_output)
        hrv_output = self.con_6(hrv_output)
        hrv_output = self.relu(hrv_output)
        hrv_output = self.con_7(hrv_output)
        hrv_output = self.relu(hrv_output)
        hrv_output = hrv_output.view(-1, num_flat_features(hrv_output))

        hrv_output = self.cls_relu(self.fc1(hrv_output))
        hrv_output = self.drop_out(hrv_output)
        hrv_output = self.cls_relu(self.fc2(hrv_output))
        hrv_output = self.drop_out(hrv_output)
        hrv_output = self.cls_relu(self.fc3(hrv_output))
        hrv_output = self.drop_out(hrv_output)
        hrv_output = self.dense(hrv_output)
        return hrv_output


class ResPlus_Raw_Appl_1(nn.Module):
    def __init__(self, in_channels, raw_acc_in_ch, num_classes, fc_dim):
        """
        this is the ResNet model for the early stage fusion strategy, which use the raw accelerometer data. We extract the
        representation from the raw accelerometer then concatenate it with the HRV features for feature learning.
        @param in_channels: the input channel
        @param raw_acc_in_ch: 3, three axis as the input channel
        @param num_classes: 3, three-stage sleep classification
        @param fc_dim: the flattened representation dimension.
        """
        super(ResPlus_Raw_Appl_1, self).__init__()

        self.con_1 = nn.Conv1d(in_channels, 512, kernel_size=3, padding=1)
        self.con_2 = nn.Conv1d(512, 512, kernel_size=3, padding=1)

        self.con_3 = nn.Conv1d(512, 128, kernel_size=3, padding=1)
        self.con_4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)

        self.con_5 = nn.Conv1d(128, 512, kernel_size=3, padding=1)
        self.con_6 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.con_7 = nn.Conv1d(512, 512, kernel_size=3, padding=1)

        self.raw_conv_1 = nn.Conv1d(raw_acc_in_ch, 512, kernel_size=3, stride=3)
        self.raw_conv_2 = nn.Conv1d(512, 512, kernel_size=3, stride=3)
        self.raw_conv_3 = nn.Conv1d(512, 512, kernel_size=3, stride=3)

        self.raw_conv_4 = nn.Conv1d(512, 128, kernel_size=12, stride=1)
        self.raw_conv_5 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.raw_conv_6 = nn.Conv1d(128, 1, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(fc_dim * 512, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.dense = nn.Linear(32, num_classes)

        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.cls_relu = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.25)
        self.raw_drop_out = nn.Dropout()

    def forward(self, acc_x, hrv_x):
        """
        @param acc_x: raw accelerometer data
        @param hrv_x: hrv features
        @return: logits and features
        """
        acc_output = self.relu(self.raw_conv_1(acc_x))
        acc_output = self.relu(self.raw_conv_2(acc_output))
        acc_output = self.raw_conv_3(acc_output)
        acc_output = self.relu(acc_output)
        acc_output = self.raw_drop_out(acc_output)
        acc_output = self.relu(self.raw_conv_4(acc_output))
        acc_output = self.relu(self.raw_conv_5(acc_output))
        acc_output = self.raw_conv_6(acc_output)
        acc_output = self.relu(acc_output)
        acc_output = self.raw_drop_out(acc_output)

        hrv_output = torch.cat((acc_output, hrv_x), dim=1)
        hrv_output = self.con_1(hrv_output)
        hrv_output = self.relu(hrv_output)
        hrv_res = hrv_output
        hrv_output = self.con_2(hrv_output)
        hrv_output += hrv_res
        hrv_output = self.relu(hrv_output)
        hrv_output = self.max_pool(hrv_output)

        hrv_output = self.con_3(hrv_output)
        hrv_res = hrv_output
        hrv_output = self.relu(hrv_output)
        hrv_output = self.con_4(hrv_output)
        hrv_output += hrv_res
        hrv_output = self.relu(hrv_output)
        hrv_output = self.max_pool(hrv_output)

        hrv_output = self.con_5(hrv_output)
        hrv_output = self.relu(hrv_output)
        hrv_res = hrv_output
        hrv_output = self.con_6(hrv_output)
        hrv_output = self.relu(hrv_output)
        hrv_output = self.con_7(hrv_output)
        hrv_output += hrv_res
        hrv_output = self.relu(hrv_output)
        hrv_output = hrv_output.view(-1, num_flat_features(hrv_output))

        hrv_output = self.cls_relu(self.fc1(hrv_output))
        hrv_output = self.drop_out(hrv_output)
        hrv_output = self.cls_relu(self.fc2(hrv_output))
        hrv_output = self.drop_out(hrv_output)
        hrv_output = self.cls_relu(self.fc3(hrv_output))
        hrv_output = self.drop_out(hrv_output)
        hrv_output = self.dense(hrv_output)
        return hrv_output


class RawAccFeatureExtraction(nn.Module):
    def __init__(self, raw_acc_in_ch):
        """
        This is the Vgg backbone network for the feature extraction of raw accelerometer
        @param raw_acc_in_ch: 3, three axes
        """
        super(RawAccFeatureExtraction, self).__init__()
        self.raw_conv_1 = nn.Conv1d(raw_acc_in_ch, 512, kernel_size=3, stride=3)
        self.raw_conv_2 = nn.Conv1d(512, 512, kernel_size=3, stride=3)
        self.raw_conv_3 = nn.Conv1d(512, 512, kernel_size=3, stride=3)

        self.raw_conv_4 = nn.Conv1d(512, 128, kernel_size=3, stride=2)
        self.raw_conv_5 = nn.Conv1d(128, 128, kernel_size=3, stride=2)
        self.raw_conv_6 = nn.Conv1d(128, 512, kernel_size=3, stride=1)

        self.relu = nn.ReLU()
        self.raw_drop_out = nn.Dropout()

    def forward(self, acc_x):
        """
        @param acc_x: raw accelerometer data
        @return: logits and features
        """
        acc_output = self.relu(self.raw_conv_1(acc_x))
        acc_output = self.relu(self.raw_conv_2(acc_output))
        acc_output = self.relu(self.raw_conv_3(acc_output))
        acc_output = self.raw_drop_out(acc_output)

        acc_output = self.relu(self.raw_conv_4(acc_output))
        acc_output = self.relu(self.raw_conv_5(acc_output))
        acc_output = self.relu(self.raw_conv_6(acc_output))
        acc_output = self.raw_drop_out(acc_output)
        return acc_output


class VggRawSplitModal(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, num_classes, fc_dim=25):
        """
        This the hybrid fusion using the concatenation method.
        @param in_channels_1: raw accelerometer input channels, e.g., three axes
        @param in_channels_2: number of HRV features,
        @param num_classes: 3, three stages
        @param fc_dim: the flattened representation dimension.
        """
        super(VggRawSplitModal, self).__init__()
        self.stream_1 = RawAccFeatureExtraction(in_channels_1)  # act
        self.stream_2 = VggAcc79F174NOFCRM(in_channels_2)  # hrv
        self.fc1 = nn.Linear(fc_dim * 512 * 2, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.dense = nn.Linear(32, num_classes)

        self.relu = nn.ReLU()
        self.cls_relu = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.25)

    def forward(self, x_1, x_2):
        """
        @param x_1: raw accelerometer data
        @param x_2: hrv features
        @return: logits and features
        """
        output_1 = self.stream_1(x_1)
        output_2 = self.stream_2(x_2)
        concat_output = torch.cat((output_1, output_2), dim=1)
        feature = concat_output.view(-1, num_flat_features(concat_output))
        feature = self.cls_relu(self.fc1(feature))
        feature = self.drop_out(feature)
        feature = self.cls_relu(self.fc2(feature))
        feature = self.drop_out(feature)
        feature = self.cls_relu(self.fc3(feature))
        feature = self.drop_out(feature)
        output = self.dense(feature)
        return feature, output


class VggRawSplitModalAdd(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, num_classes, fc_dim=25):
        """
        This is the hybrid fusion strategy using addition method based on the raw accelerometer data and
        the HRV features.
        @param in_channels_1: raw accelerometer input channels, e.g., three axes
        @param in_channels_2: number of HRV features.
        @param num_classes: 3, three stages.
        @param fc_dim: the flattened representation dimension.
        """
        super(VggRawSplitModalAdd, self).__init__()
        self.stream_1 = RawAccFeatureExtraction(in_channels_1)  # act
        self.stream_2 = VggAcc79F174NOFCRM(in_channels_2)  # hrv
        self.fc1 = nn.Linear(fc_dim * 512, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.dense = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.cls_relu = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.25)

    def forward(self, x_1, x_2):
        """
        @param x_1: raw accelerometer data
        @param x_2: hrv features
        @return: logits and features
        """
        output_1 = self.stream_1(x_1)
        output_2 = self.stream_2(x_2)

        concat_output = output_1+output_2
        feature = concat_output.view(-1, num_flat_features(concat_output))
        feature = self.cls_relu(self.fc1(feature))
        feature = self.drop_out(feature)
        feature = self.cls_relu(self.fc2(feature))
        feature = self.drop_out(feature)
        feature = self.cls_relu(self.fc3(feature))
        feature = self.drop_out(feature)
        output = self.dense(feature)
        return feature, output


class VggRawSANTiDimMatAttMod1NLayer1Con(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, num_classes, att_on_modality, time_step_dim=5):
        """
        This is the hybrid fusion strategy using addition method based on the raw accelerometer data and
        the HRV features.

        @param in_channels_1: raw accelerometer input channels, e.g., three axes
        @param in_channels_2: number of HRV features.
        @param num_classes: 3, three stages.
        @param att_on_modality: which modality should be focused
        @param time_step_dim: total temporal steps in the attention operation
        """
        super(VggRawSANTiDimMatAttMod1NLayer1Con, self).__init__()
        self.stream_1 = RawAccFeatureExtraction(in_channels_1)  # act
        self.stream_2 = VggAcc79F174NOFCRM(in_channels_2)  # hrv
        self.san = SANTimeDimMatrixAttOnModality1NLayer1Concat(input_feature_dim=512, attention_dim=256,
                                                               time_step_dim=time_step_dim)
        self.fc1 = nn.Linear(time_step_dim * 512 * 2, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.dense = nn.Linear(32, num_classes)
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.cls_relu = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.25)
        assert att_on_modality in ["act", "car"]
        self.att_on_modality = att_on_modality

    def forward(self, x_1, x_2):
        """
        @param x_1: raw accelerometer data
        @param x_2: hrv features
        @return: logits and features
        """
        output_1 = self.stream_1(x_1)
        output_2 = self.stream_2(x_2)
        if self.att_on_modality == "act":
            att_output = self.san(output_1, output_2)
        else:
            att_output = self.san(output_2, output_1)
        feature = att_output.view(-1, num_flat_features(att_output))
        feature = self.cls_relu(self.fc1(feature))
        feature = self.drop_out(feature)
        feature = self.cls_relu(self.fc2(feature))
        feature = self.drop_out(feature)
        feature = self.cls_relu(self.fc3(feature))
        feature = self.drop_out(feature)
        output = self.dense(feature)
        return feature, output


class ResPlusRawSplitModalCon(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, num_classes, time_step_dim=25):
        """
        This is the ResNet model in hybrid fusion strategy using the concatenation method. It intakes raw accelerometer
        data and HRV features as the inputs.
        @param in_channels_1: raw accelerometer input channels, e.g., three axes
        @param in_channels_2: number of HRV features.
        @param num_classes: 3, three-stages
        @param time_step_dim: total temporal steps in the attention operation
        """
        super(ResPlusRawSplitModalCon, self).__init__()
        self.stream_1 = RawAccFeatureExtraction(in_channels_1)  # act
        self.stream_2 = ResPlusSingleModalityNOFC(in_channels_2)  # hrv
        self.fc1 = nn.Linear(time_step_dim * 512 * 2, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.dense = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.cls_relu = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.25)

    def forward(self, x_1, x_2):
        """
        @param x_1: raw accelerometer data
        @param x_2: hrv features
        @return: logits and features
        """
        output_1 = self.stream_1(x_1)
        output_2 = self.stream_2(x_2)
        concat_output = torch.cat((output_1, output_2), dim=1)
        feature = concat_output.view(-1, num_flat_features(concat_output))
        feature = self.cls_relu(self.fc1(feature))
        feature = self.drop_out(feature)
        feature = self.cls_relu(self.fc2(feature))
        feature = self.drop_out(feature)
        feature = self.cls_relu(self.fc3(feature))
        feature = self.drop_out(feature)
        output = self.dense(feature)
        return feature, output


class ResPlusRawSplitModalPlus(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, num_classes, time_step_dim=25):
        """
        This is the ResNet model in the hybrid fusion strategy using the addition method. It intakes the raw
        accelerometer data and the HRV data.
        @param in_channels_1: accelerometer data axes
        @param in_channels_2: HRV data input channels
        @param num_classes: 3, three stages
        @param time_step_dim:  total temporal steps in the attention operation
        """
        super(ResPlusRawSplitModalPlus, self).__init__()
        self.stream_1 = RawAccFeatureExtraction(in_channels_1)  # act
        self.stream_2 = ResPlusSingleModalityNOFC(in_channels_2)  # hrv
        self.fc1 = nn.Linear(time_step_dim * 512, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.dense = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.cls_relu = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.25)

    def forward(self, x_1, x_2):
        """
        @param x_1: raw accelerometer data
        @param x_2: hrv features
        @return: logits and features
        """
        output_1 = self.stream_1(x_1)
        output_2 = self.stream_2(x_2)

        concat_output = output_1 + output_2
        feature = concat_output.view(-1, num_flat_features(concat_output))
        feature = self.cls_relu(self.fc1(feature))
        feature = self.drop_out(feature)
        feature = self.cls_relu(self.fc2(feature))
        feature = self.drop_out(feature)
        feature = self.cls_relu(self.fc3(feature))
        feature = self.drop_out(feature)
        output = self.dense(feature)
        return feature, output


class ResRawPlusSplitModal_SANTiDimMatAttMod1NLayer1Con(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, num_classes, att_on_modality, time_step_dim=25):
        """
        This is the method in hybrid fusion strategy using the attention method. It takes the raw accelerometer data,
        and HRV features as the model inputs.
        @param in_channels_1: accelerometer axes
        @param in_channels_2: the total number of channels of HRV features
        @param num_classes: 3, three sleep stages
        @param att_on_modality: specify which modality should the network focus on
        @param time_step_dim: total temporal steps.
        """
        super(ResRawPlusSplitModal_SANTiDimMatAttMod1NLayer1Con, self).__init__()
        self.stream_1 = RawAccFeatureExtraction(in_channels_1)  # act
        self.stream_2 = ResPlusSingleModalityNOFC(in_channels_2)  # hrv
        self.san = SANTimeDimMatrixAttOnModality1NLayer1Concat(input_feature_dim=512, attention_dim=256,
                                                               time_step_dim=time_step_dim)
        self.fc1 = nn.Linear(time_step_dim * 512 * 2, 512)  # the concatenation will double 512
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.dense = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.cls_relu = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.25)
        assert att_on_modality in ["act", "car"]
        self.att_on_modality = att_on_modality

    def forward(self, x_1, x_2):
        """
        @param x_1: raw accelerometer data
        @param x_2: hrv features
        @return: logits and features
        """
        output_1 = self.stream_1(x_1)
        output_2 = self.stream_2(x_2)
        if self.att_on_modality == "act":
            att_output = self.san(output_1, output_2)
        else:
            att_output = self.san(output_2, output_1)
        feature = att_output.view(-1, num_flat_features(att_output))
        feature = self.cls_relu(self.fc1(feature))
        feature = self.drop_out(feature)
        feature = self.cls_relu(self.fc2(feature))
        feature = self.drop_out(feature)
        feature = self.cls_relu(self.fc3(feature))
        feature = self.drop_out(feature)
        output = self.dense(feature)
        return feature, output


class ResPlusRawSplitModal_BiLinear(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, num_classes):
        """
        This is the ResNet model in hybrid fusion strategy using the bilinear method. It takes raw accelerometer data
        and HRV features as the inputs.
        @param in_channels_1: total number of accelerometer axes
        @param in_channels_2: total number of channels of HRV features
        @param num_classes: 3 , three sleep stages
        """
        super(ResPlusRawSplitModal_BiLinear, self).__init__()
        self.stream_1 = RawAccFeatureExtraction(in_channels_1)
        self.stream_2 = ResPlusSingleModalityNOFC(in_channels_2)
        self.bl = BilinearFusion(512, 1024)
        self.fc1 = nn.Linear(1024, 512)  # bilinear output is 1024
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.dense = nn.Linear(32, num_classes)
        # self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.cls_relu = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.25)

    def forward(self, x_1, x_2):
        """
        @param x_1: raw accelerometer data
        @param x_2: hrv features
        @return: logits and features
        """
        output_1 = self.stream_1(x_1)
        output_2 = self.stream_2(x_2)

        att_output = self.bl(output_1, output_2)
        feature = self.cls_relu(self.fc1(att_output))
        feature = self.drop_out(feature)
        feature = self.cls_relu(self.fc2(feature))
        feature = self.drop_out(feature)
        feature = self.cls_relu(self.fc3(feature))
        feature = self.drop_out(feature)
        output = self.dense(feature)
        return feature, output


class VggRaw_BiLinear(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, num_classes):
        """
        This is the Vgg model for raw accelerometer data and HRV features in late stage fusion using
        the Bilinear fusion method
        @param in_channels_1: total number of accelerometer axes
        @param in_channels_2: total number of channels of HRV features
        @param num_classes: 3 , three sleep stages
        """
        super(VggRaw_BiLinear, self).__init__()
        self.stream_1 = RawAccFeatureExtraction(in_channels_1)  # act
        self.stream_2 = VggAcc79F174NOFCRM(in_channels_2)  # hrv
        self.bl = BilinearFusion(512, 1024)
        self.fc1 = nn.Linear(2 * 512, 512)  # was 12 x 512 * 2 now is 12 x 512
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.dense = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.cls_relu = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.25)

    def forward(self, x_1, x_2):
        """
        @param x_1: raw accelerometer data
        @param x_2: hrv features
        @return: logits and features
        """
        output_1 = self.stream_1(x_1)
        output_2 = self.stream_2(x_2)
        att_output = self.bl(output_1, output_2)
        feature = self.cls_relu(self.fc1(att_output))
        feature = self.drop_out(feature)
        feature = self.cls_relu(self.fc2(feature))
        feature = self.drop_out(feature)
        feature = self.cls_relu(self.fc3(feature))
        feature = self.drop_out(feature)
        output = self.dense(feature)
        return feature, output


class VggRaw2DConcate(nn.Module):
    def __init__(self, in_channels_1, num_classes, fc_dim=25):
        """
        This is the Vgg model for raw accelerometer data and HRV features in late stage fusion using the concatenation
        method.
        @param in_channels_1: the accelerometer data axes
        @param num_classes: 3, three sleep stages
        @param fc_dim: the flattened representation dimension.
        """
        super(VggRaw2DConcate, self).__init__()
        self.stream_1 = RawAccFeatureExtraction(in_channels_1)  # act
        self.stream_2 = VggIMGNOFC()  # hrv
        self.fc1 = nn.Linear(fc_dim * 512 * 7, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.dense = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.cls_relu = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.25)

    def forward(self, x_1, x_2):
        """
        @param x_1: raw accelerometer data
        @param x_2: hrv features
        @return: logits and features
        """
        output_1 = self.stream_1(x_1)
        x_2 = torch.unsqueeze(x_2, dim=1)
        output_2 = self.stream_2(x_2)
        output_1 = torch.unsqueeze(output_1, dim=2)
        concat_output = torch.cat((output_1, output_2), dim=2)
        feature = concat_output.view(-1, num_flat_features(concat_output))
        feature = self.cls_relu(self.fc1(feature))
        feature = self.drop_out(feature)
        feature = self.cls_relu(self.fc2(feature))
        feature = self.drop_out(feature)
        feature = self.cls_relu(self.fc3(feature))
        feature = self.drop_out(feature)
        output = self.dense(feature)
        return feature, output


class VggRaw2DSum(nn.Module):
    def __init__(self, in_channels_1, num_classes, fc_dim=25):
        """
        This is the Vgg model for raw accelerometer data and HRV features in late stage fusion using the addition
        method.
        @param in_channels_1: the accelerometer data axes
        @param num_classes: 3, three sleep stages
        @param fc_dim: the flattened representation dimension.
        """
        super(VggRaw2DSum, self).__init__()
        self.stream_1 = RawAccFeatureExtraction(in_channels_1)  # act
        self.stream_2 = VggIMGNOFC()  # hrv
        self.fc1 = nn.Linear(fc_dim * 512, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.dense = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.cls_relu = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.25)

    def forward(self, x_1, x_2):
        """
        @param x_1: raw accelerometer data
        @param x_2: hrv features
        @return: logits and features
        """
        output_1 = self.stream_1(x_1)
        x_2 = torch.unsqueeze(x_2, dim=1)
        output_2 = self.stream_2(x_2)
        output_1 = torch.unsqueeze(output_1, dim=2)
        concat_output = torch.cat((output_1, output_2), dim=2)
        concat_output = concat_output.sum(axis=2)
        feature = concat_output.view(-1, num_flat_features(concat_output))
        feature = self.cls_relu(self.fc1(feature))
        feature = self.drop_out(feature)
        feature = self.cls_relu(self.fc2(feature))
        feature = self.drop_out(feature)
        feature = self.cls_relu(self.fc3(feature))
        feature = self.drop_out(feature)
        output = self.dense(feature)
        return feature, output


class VggRaw2DResConcate(nn.Module):
    def __init__(self, in_channels_1, num_classes, fc_dim=25):
        """
        This is the Vgg model for raw accelerometer data and HRV features in late stage fusion using the concatenation
        method.
        @param in_channels_1: the accelerometer data axes.
        @param num_classes: 3, three sleep stages.
        @param fc_dim: the flattened representation dimension.
        """
        super(VggRaw2DResConcate, self).__init__()
        self.stream_1 = RawAccFeatureExtraction(in_channels_1)  # act
        self.stream_2 = VggIMGResNOFC()  # hrv
        self.fc1 = nn.Linear(fc_dim * 512 * 7, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.dense = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.cls_relu = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.25)

    def forward(self, x_1, x_2):
        """
        @param x_1: raw accelerometer data
        @param x_2: hrv features
        @return: logits and features
        """
        output_1 = self.stream_1(x_1)
        x_2 = torch.unsqueeze(x_2, dim=1)
        output_2 = self.stream_2(x_2)
        output_1 = torch.unsqueeze(output_1, dim=2)
        concat_output = torch.cat((output_1, output_2), dim=2)
        feature = concat_output.view(-1, num_flat_features(concat_output))
        feature = self.cls_relu(self.fc1(feature))
        feature = self.drop_out(feature)
        feature = self.cls_relu(self.fc2(feature))
        feature = self.drop_out(feature)
        feature = self.cls_relu(self.fc3(feature))
        feature = self.drop_out(feature)
        output = self.dense(feature)
        return feature, output


class VggRaw2DResSum(nn.Module):
    def __init__(self, in_channels_1, num_classes, fc_dim=25):
        """
        This is the ResNet model for raw accelerometer data and HRV features in late stage fusion using the addition
        method.
        @param in_channels_1: the accelerometer data axes.
        @param num_classes: 3, three sleep stages.
        @param fc_dim: the flattened representation dimension.
        """
        super(VggRaw2DResSum, self).__init__()
        self.stream_1 = RawAccFeatureExtraction(in_channels_1)  # act
        self.stream_2 = VggIMGResNOFC()  # hrv
        self.fc1 = nn.Linear(fc_dim * 512, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.dense = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.cls_relu = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.25)

    def forward(self, x_1, x_2):
        """
        @param x_1: raw accelerometer data
        @param x_2: hrv features
        @return: logits and features
        """
        output_1 = self.stream_1(x_1)
        x_2 = torch.unsqueeze(x_2, dim=1)
        output_2 = self.stream_2(x_2)
        output_1 = torch.unsqueeze(output_1, dim=2)
        concat_output = torch.cat((output_1, output_2), dim=2)
        concat_output = concat_output.sum(axis=2)
        feature = concat_output.view(-1, num_flat_features(concat_output))

        feature = self.cls_relu(self.fc1(feature))
        feature = self.drop_out(feature)
        feature = self.cls_relu(self.fc2(feature))
        feature = self.drop_out(feature)
        feature = self.cls_relu(self.fc3(feature))
        feature = self.drop_out(feature)
        output = self.dense(feature)
        return feature, output

