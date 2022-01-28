from models.mix_2dmodel import *
from models.build_model import get_model_time_step_dim


def get_num_in_channel(dataset_name="mesa"):
    if dataset_name == "mesa":
        in_channel = 9
    elif dataset_name == "apple":
        in_channel = 7
    elif dataset_name == "mesa_hr_statistic":
        in_channel = 7
    else:
        raise ValueError("Sorry, dataset is not recognised should be mesa, mesa_hr_statistic, apple")
    return in_channel


def get_num_in_channel_1_2(dataset_name="mesa"):
    if dataset_name == "mesa":
        in_channel_1 = 1
        in_channel_2 = 8
    elif dataset_name == "apple":
        in_channel_1 = 1
        in_channel_2 = 6
    elif dataset_name == "mesa_hr_statistic":
        in_channel_1 = 1
        in_channel_2 = 6
    else:
        raise ValueError("Sorry, dataset is not recognised should be mesa, mesa_hr_statistic, apple")
    return in_channel_1, in_channel_2


def build_2d_model(nn_type, dataset, num_classes, seq_len):
    if nn_type == "VggIMG":
        sleep_model = VggIMG(in_channels=get_num_in_channel(dataset), num_classes=num_classes,
                             time_step_dim=get_model_time_step_dim(nn_type, seq_len))
    elif nn_type == "VggIMGSum":
        sleep_model = VggIMGSum(in_channels=get_num_in_channel(dataset), num_classes=num_classes,
                                time_step_dim=get_model_time_step_dim(nn_type, seq_len))
    elif nn_type == "VggIMGRes":
        sleep_model = VggIMGRes(in_channels=get_num_in_channel(dataset), num_classes=num_classes,
                                time_step_dim=get_model_time_step_dim(nn_type, seq_len))
    elif nn_type == "VggIMGResSum":
        sleep_model = VggIMGResSum(in_channels=get_num_in_channel(dataset), num_classes=num_classes,
                                   time_step_dim=get_model_time_step_dim(nn_type, seq_len))
    else:
        raise Exception("model specified not existed!")
    return sleep_model
