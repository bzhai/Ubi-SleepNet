from models.baseline_models import *
from models.raw_data_models import *

import yaml
import os
from easydict import EasyDict as edict
from utilities.utils import get_project_root


def get_model_time_step_dim(model_name, seq_len):
    with open(os.path.join(get_project_root(), "model_settings.yaml")) as f:
        exp_config = edict(yaml.load(f))
    fc_dim = exp_config[model_name].FC_TEMP_STEPS[str(seq_len)]
    return fc_dim


def get_num_in_channel_raw_acc_hr(dataset_name="mesa"):
    if dataset_name == "mesa_raw":
        in_channel_1 = 2
        in_channel_2 = 0
    elif dataset_name == "apple_raw":
        in_channel_1 = 1
        in_channel_2 = 3
    else:
        raise ValueError("Sorry, dataset is not recognised should be mesa, mesa_hr_statistic, apple")
    return in_channel_1, in_channel_2


def get_fc_dim_raw(win_len):
    if win_len == 100:
        fc_dimension = 25
    elif win_len == 50:
        fc_dimension = 12
    elif win_len == 20:
        fc_dimension = 4
    else:
        raise Exception("win_len is incorrect")
    return fc_dimension


def build_raw_acc_hr_model(nn_type, dataset, seq_len, num_classes, modality="none"):
    if (nn_type == "VggAcc79F174_7_RM_Raw_Appl_1_hr") & (dataset == "apple_raw"):
        model = VggAcc79F174_7_RM_Raw_Appl_1(in_channels=get_num_in_channel_raw_acc_hr(dataset)[0] + 1,
                                             raw_acc_in_ch=get_num_in_channel_raw_acc_hr(dataset)[1], num_classes=num_classes,
                                             fc_dim=get_fc_dim_raw(seq_len))
    elif (nn_type == "ResPlus_Raw_Appl_1_hr") & (dataset == "apple_raw"):
        model = ResPlus_Raw_Appl_1(in_channels=get_num_in_channel_raw_acc_hr(dataset)[0] + 1,
                                   raw_acc_in_ch=get_num_in_channel_raw_acc_hr(dataset)[1], num_classes=num_classes,
                                   fc_dim=get_fc_dim_raw(seq_len))

    elif (nn_type == "VggRawSplitModal_hr") & (dataset == "apple_raw"):
        model = VggRawSplitModal(in_channels_1=get_num_in_channel_raw_acc_hr(dataset)[1],
                                 in_channels_2=get_num_in_channel_raw_acc_hr(dataset)[0], num_classes=num_classes,
                                 fc_dim=get_fc_dim_raw(seq_len))
    elif (nn_type == "VggRawSplitModalAdd_hr") & (dataset == "apple_raw"):
        model = VggRawSplitModalAdd(in_channels_1=get_num_in_channel_raw_acc_hr(dataset)[1],
                                    in_channels_2=get_num_in_channel_raw_acc_hr(dataset)[0], num_classes=num_classes,
                                    fc_dim=get_fc_dim_raw(seq_len))
    elif (nn_type == "VggRawSANTiDimMatAttMod1NLayer1Con_hr") & (dataset == "apple_raw"):
        assert modality != "none"
        model = VggRawSANTiDimMatAttMod1NLayer1Con(
            in_channels_1=get_num_in_channel_raw_acc_hr(dataset)[1],
            in_channels_2=get_num_in_channel_raw_acc_hr(dataset)[0],
            num_classes=num_classes,
            att_on_modality=modality,
            time_step_dim=get_model_time_step_dim(nn_type, seq_len)
        )
    elif nn_type == "ResPlusRawSplitModalCon_hr":
        model = ResPlusRawSplitModalCon(in_channels_1=get_num_in_channel_raw_acc_hr(dataset)[1],
                                        in_channels_2=get_num_in_channel_raw_acc_hr(dataset)[0],
                                        num_classes=num_classes,
                                        time_step_dim=get_model_time_step_dim(nn_type, seq_len)
                                        )
    elif nn_type == "ResPlusRawSplitModalPlus_hr":
        model = ResPlusRawSplitModalPlus(in_channels_1=get_num_in_channel_raw_acc_hr(dataset)[1],
                                         in_channels_2=get_num_in_channel_raw_acc_hr(dataset)[0],
                                         num_classes=num_classes,
                                         time_step_dim=get_model_time_step_dim(nn_type, seq_len)
                                         )
    elif nn_type == "ResRawPlusSplitModal_SANTiDimMatAttMod1NLayer1Con_hr":
        assert modality != "none"
        model = ResRawPlusSplitModal_SANTiDimMatAttMod1NLayer1Con(
            in_channels_1=get_num_in_channel_raw_acc_hr(dataset)[1],
            in_channels_2=get_num_in_channel_raw_acc_hr(dataset)[0],
            num_classes=num_classes,
            att_on_modality=modality,
            time_step_dim=get_model_time_step_dim(nn_type, seq_len=seq_len)
        )

    elif nn_type == "ResPlusRawSplitModal_BiLinear_hr":
        model = ResPlusRawSplitModal_BiLinear(
            in_channels_1=get_num_in_channel_raw_acc_hr(dataset)[1],
            in_channels_2=get_num_in_channel_raw_acc_hr(dataset)[0],
            num_classes=num_classes,
        )

    elif nn_type == "VggRaw_BiLinear_hr":
        model = VggRaw_BiLinear(
            in_channels_1=get_num_in_channel_raw_acc_hr(dataset)[1],
            in_channels_2=get_num_in_channel_raw_acc_hr(dataset)[0],
            num_classes=num_classes,
        )
    else:
        raise Exception("model specified not existed!")
    return model


if __name__ == "__main__":
    get_model_time_step_dim('VggAcc79F174_RM_SANTiDimMatAttMod1NLayer1Con', 21)
