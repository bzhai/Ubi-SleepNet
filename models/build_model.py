from models.baseline_models import *
from models.mix_model import *
from models.removed_last_maxpool import *
import yaml
import os
from easydict import EasyDict as edict
from utilities.utils import get_project_root


def get_model_time_step_dim(model_name, seq_len):
    """
    return the last feature map's temporal time steps.
    @param model_name: model name
    @param seq_len: sliding window length e.g., 20, 50, 100.
    @return:
    """
    with open(os.path.join(get_project_root(), "model_settings.yaml")) as f:
        exp_config = edict(yaml.load(f, yaml.Loader))
    fc_dim = exp_config[model_name].FC_TEMP_STEPS[str(seq_len)]
    return fc_dim


def get_num_in_channel(dataset_name="mesa"):
    """
    get the number of input channels, this is for early stage fusion as they use Con1D
    @param dataset_name: the dataset name
    @return: number of channels
    """
    if dataset_name == "mesa":
        in_channel = 9
    elif dataset_name == "mesa_loocv":
        in_channel = 7
    elif dataset_name == "apple":
        in_channel = 7
    elif dataset_name == "mesa_hr_statistic":
        in_channel = 7
    else:
        raise ValueError("Sorry, dataset is not recognised should be mesa, mesa_hr_statistic, apple")
    return in_channel


def get_num_in_channel_2D(dataset_name="mesa"):
    """
    get the number of input channels, this is for the hybrid fusion strategy as they use two Conv1D networks.
    @param dataset_name: dataset name
    @return: a set, one for each network
    """
    if dataset_name == "mesa":
        in_channel_1 = 1
        in_channel_2 = 8
    elif dataset_name == "mesa_loocv":
        in_channel_1 = 1
        in_channel_2 = 6
    elif dataset_name == "apple":
        in_channel_1 = 1
        in_channel_2 = 6

    elif dataset_name == "mesa_hr_statistic":
        in_channel_1 = 1
        in_channel_2 = 6
    else:
        raise ValueError("Sorry, dataset is incorrect should be mesa, mesa_hr_statistic, apple")
    return in_channel_1, in_channel_2


def build_model(nn_type, dataset, num_classes, seq_len, modality):
    """
    get a model for fusion experiments, this function only covers early stage fusion and hybrid fusion.
    @param nn_type: neural network name
    @param dataset: dataset name
    @param num_classes: only use 3 classes
    @param seq_len: ava sequence lengths are 20, 50, 101
    @param modality: only available for attention models "act, car, none"
    @return: a pytorch model for fusion.
    """
    if nn_type == "CNN":
        model = CNN(in_channels=get_num_in_channel(dataset), num_classes=num_classes)
    elif nn_type == "VggAcc79F174ResdPlus":
        model = VggAcc79F174ResdPlus(in_channels=get_num_in_channel(dataset), num_classes=num_classes,
                                     time_step_dim=get_model_time_step_dim(nn_type, seq_len))
    elif nn_type == "ResPlusSplitModalCon":
        model = ResPlusSplitModalCon(in_channels_1=get_num_in_channel_2D(dataset)[0],
                                     in_channels_2=get_num_in_channel_2D(dataset)[1],
                                     num_classes=num_classes,
                                     time_step_dim=get_model_time_step_dim(nn_type, seq_len)
                                     )
    elif nn_type == "ResPlusSplitModalPlus":
        model = ResPlusSplitModalPlus(
            in_channels_1=get_num_in_channel_2D(dataset)[0],
            in_channels_2=get_num_in_channel_2D(dataset)[1],
            num_classes=num_classes,
            time_step_dim=get_model_time_step_dim(nn_type, seq_len)
        )

    # ################ ResNet as ResNet use plus ###############
    elif nn_type == "ResPlusSplitModal_BiLinear":
        model = ResPlusSplitModal_BiLinear(
            in_channels_1=get_num_in_channel_2D(dataset)[0],
            in_channels_2=get_num_in_channel_2D(dataset)[1],
            num_classes=num_classes
        )

    elif nn_type == "ResPlusSplitModal_SANTiDimMatAttMod1NLayer1Con":
        assert modality != "none"
        model = ResPlusSplitModal_SANTiDimMatAttMod1NLayer1Con(
            in_channels_1=get_num_in_channel_2D(dataset)[0],
            in_channels_2=get_num_in_channel_2D(dataset)[1],
            num_classes=num_classes,
            att_on_modality=modality,
            time_step_dim=get_model_time_step_dim('ResPlusSplitModal_SANTiDimMatAttMod1NLayer1Con', seq_len=seq_len)
        )
    # ############### change the VGG network removed the last max pooling layer ###############
    elif nn_type == "VggAcc79F174_7_RM":
        model = VggAcc79F174_7_RM(in_channels=get_num_in_channel(dataset), num_classes=num_classes,
                                  fc_dim=get_model_time_step_dim(nn_type, seq_len))
    elif nn_type == "VggAcc79F174_RM_SplitModal":
        model = VggAcc79F174_RM_SplitModal(in_channels_1=get_num_in_channel_2D(dataset)[0],
                                           in_channels_2=get_num_in_channel_2D(dataset)[1],
                                           num_classes=num_classes,
                                           fc_dim=get_model_time_step_dim(nn_type, seq_len))
    elif nn_type == "VggAcc79F174_RM_SplitModalAdd":
        model = VggAcc79F174_RM_SplitModalAdd(in_channels_1=get_num_in_channel_2D(dataset)[0],
                                              in_channels_2=get_num_in_channel_2D(dataset)[1],
                                              num_classes=num_classes,
                                              fc_dim=get_model_time_step_dim(nn_type, seq_len))
    elif nn_type == "VggAcc79F174_RM_SANTiDimMatAttMod1NLayer1Con":
        assert modality != "none"
        model = VggAcc79F174_RM_SANTiDimMatAttMod1NLayer1Con(
            in_channels_1=get_num_in_channel_2D(dataset)[0],
            in_channels_2=get_num_in_channel_2D(dataset)[1],
            num_classes=num_classes,
            att_on_modality=modality,
            time_step_dim=get_model_time_step_dim(nn_type, seq_len)
        )
    elif nn_type == "VggAcc79F174_RM_BiLinear":
        model = VggAcc79F174_RM_BiLinear(
            in_channels_1=get_num_in_channel_2D(dataset)[0],
            in_channels_2=get_num_in_channel_2D(dataset)[1],
            num_classes=num_classes,
        )
    else:
        raise Exception("model specified not existed!")
    return model
