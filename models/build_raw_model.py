from models.baseline_models import *
from models.raw_data_models import *

import yaml
import os
from easydict import EasyDict as edict
from utilities.utils import get_project_root


def get_model_time_step_dim(model_name, seq_len):
    with open(os.path.join(get_project_root(), "model_settings.yaml")) as f:
        exp_config = edict(yaml.load(f, yaml.Loader))
    fc_dim = exp_config[model_name].FC_TEMP_STEPS[str(seq_len)]
    return fc_dim


def get_num_in_channel_raw(dataset_name="mesa"):
    if dataset_name == "mesa_raw":
        in_channel_1 = 2
        in_channel_2 = 0
    elif dataset_name == "apple_raw":
        in_channel_1 = 6
        in_channel_2 = 3
    else:
        raise ValueError("Sorry, dataset is not recognised should be mesa, mesa_hr_statistic, apple")
    return in_channel_1, in_channel_2


# def get_num_in_channel_1_2(dataset_name="mesa"):
#     if dataset_name == "mesa":
#         in_channel_1 = 1
#         in_channel_2 = 8
#     elif dataset_name == "apple":
#         in_channel_1 = 1
#         in_channel_2 = 6
#     elif dataset_name == "mesa_hr_statistic":
#         in_channel_1 = 1
#         in_channel_2 = 6
#     else:
#         raise ValueError("Sorry, dataset is not recognised should be mesa, mesa_hr_statistic, apple")
#     return in_channel_1, in_channel_2

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


def build_raw_model(nn_type, dataset, seq_len, num_classes, modality="none"):
    # if (nn_type == "VggAcc79F174_7_RM_RAW") & (dataset == "mesa_raw"):
    #     model = VggAcc79F174_7_RM_RAW_MESA(in_channels=get_num_in_channel_raw(dataset), num_classes=num_classes,
    #                                        fc_dim=get_fc_dim_raw(seq_len))
    # elif (nn_type == "VggAcc79F174_7_RM_Raw_Appl") & (dataset == "apple_raw"):
    #     model = VggAcc79F174_7_RM_Raw_Appl(in_channels=get_num_in_channel_raw(dataset)[0]*2,
    #                                        raw_acc_in_ch=get_num_in_channel_raw(dataset)[1], num_classes=num_classes,
    #                                        fc_dim=get_fc_dim_raw(seq_len))
    if (nn_type == "VggAcc79F174_7_RM_Raw_Appl_1") & (dataset == "apple_raw"):
        model = VggAcc79F174_7_RM_Raw_Appl_1(in_channels=get_num_in_channel_raw(dataset)[0]+1,
                                           raw_acc_in_ch=get_num_in_channel_raw(dataset)[1], num_classes=num_classes,
                                           fc_dim=get_fc_dim_raw(seq_len))
    elif (nn_type == "ResPlus_Raw_Appl_1") & (dataset == "apple_raw"):
        model = ResPlus_Raw_Appl_1(in_channels=get_num_in_channel_raw(dataset)[0] + 1,
                                   raw_acc_in_ch=get_num_in_channel_raw(dataset)[1], num_classes=num_classes,
                                   fc_dim=get_fc_dim_raw(seq_len))

    # elif (nn_type == "ResPlus_Raw_Appl") & (dataset == "apple_raw"):
    #     model = ResPlus_Raw_Appl(in_channels=get_num_in_channel_raw(dataset)[0]*2,
    #                                        raw_acc_in_ch=get_num_in_channel_raw(dataset)[1], num_classes=num_classes,
    #                                        fc_dim=get_fc_dim_raw(seq_len))
    elif (nn_type == "VggRawSplitModal") & (dataset == "apple_raw"):
        model = VggRawSplitModal(in_channels_1=get_num_in_channel_raw(dataset)[1],
                                 in_channels_2=get_num_in_channel_raw(dataset)[0], num_classes=num_classes,
                                 fc_dim=get_fc_dim_raw(seq_len))
    elif (nn_type == "VggRawSplitModalAdd") & (dataset == "apple_raw"):
        model = VggRawSplitModalAdd(in_channels_1=get_num_in_channel_raw(dataset)[1],
                                 in_channels_2=get_num_in_channel_raw(dataset)[0], num_classes=num_classes,
                                 fc_dim=get_fc_dim_raw(seq_len))
    elif (nn_type == "VggRawSANTiDimMatAttMod1NLayer1Con") & (dataset == "apple_raw"):
        assert modality != "none"
        model = VggRawSANTiDimMatAttMod1NLayer1Con(
            in_channels_1=get_num_in_channel_raw(dataset)[1],
            in_channels_2=get_num_in_channel_raw(dataset)[0],
            num_classes=num_classes,
            att_on_modality=modality,
            time_step_dim=get_model_time_step_dim(nn_type, seq_len)
        )
    elif nn_type == "ResPlusRawSplitModalCon":
        model = ResPlusRawSplitModalCon(in_channels_1=get_num_in_channel_raw(dataset)[1],
                                     in_channels_2=get_num_in_channel_raw(dataset)[0],
                                     num_classes=num_classes,
                                     time_step_dim=get_model_time_step_dim(nn_type, seq_len)
                                     )
    elif nn_type == "ResPlusRawSplitModalPlus":
        model = ResPlusRawSplitModalPlus(in_channels_1=get_num_in_channel_raw(dataset)[1],
                                        in_channels_2=get_num_in_channel_raw(dataset)[0],
                                        num_classes=num_classes,
                                        time_step_dim=get_model_time_step_dim(nn_type, seq_len)
                                        )
    elif nn_type == "ResRawPlusSplitModal_SANTiDimMatAttMod1NLayer1Con":
        assert modality != "none"
        model = ResRawPlusSplitModal_SANTiDimMatAttMod1NLayer1Con(
            in_channels_1=get_num_in_channel_raw(dataset)[1],
            in_channels_2=get_num_in_channel_raw(dataset)[0],
            num_classes=num_classes,
            att_on_modality=modality,
            time_step_dim=get_model_time_step_dim('ResRawPlusSplitModal_SANTiDimMatAttMod1NLayer1Con', seq_len=seq_len)
        )

    elif nn_type == "ResPlusRawSplitModal_BiLinear":
        model = ResPlusRawSplitModal_BiLinear(
            in_channels_1=get_num_in_channel_raw(dataset)[1],
            in_channels_2=get_num_in_channel_raw(dataset)[0],
            num_classes=num_classes,
        )

    elif nn_type == "VggRaw_BiLinear":
        model = VggRaw_BiLinear(
            in_channels_1=get_num_in_channel_raw(dataset)[1],
            in_channels_2=get_num_in_channel_raw(dataset)[0],
            num_classes=num_classes,
        )
    elif (nn_type == "VggRaw2DConcate") & (dataset == "apple_raw"):
        model = VggRaw2DConcate(in_channels_1=get_num_in_channel_raw(dataset)[1],
                                 num_classes=num_classes,
                                 fc_dim=get_fc_dim_raw(seq_len))
    elif (nn_type == "VggRaw2DSum") & (dataset == "apple_raw"):
        model = VggRaw2DSum(in_channels_1=get_num_in_channel_raw(dataset)[1],
                                num_classes=num_classes,
                                fc_dim=get_fc_dim_raw(seq_len))
    elif (nn_type == "VggRaw2DResConcate") & (dataset == "apple_raw"):
        model = VggRaw2DResConcate(in_channels_1=get_num_in_channel_raw(dataset)[1],
                                num_classes=num_classes,
                                fc_dim=get_fc_dim_raw(seq_len))
    elif (nn_type == "VggRaw2DResSum") & (dataset == "apple_raw"):
        model = VggRaw2DResSum(in_channels_1=get_num_in_channel_raw(dataset)[1],
                            num_classes=num_classes,
                            fc_dim=get_fc_dim_raw(seq_len))
    else:
        raise Exception("model specified not existed!")
    return model


if __name__ == "__main__":
    get_model_time_step_dim('VggAcc79F174_RM_SANTiDimMatAttMod1NLayer1Con', 21)
