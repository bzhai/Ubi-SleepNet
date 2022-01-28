import torch

from models.build_model import build_model
from models.build_raw_model import build_raw_model
from models.mix_model import *
from models.removed_last_maxpool import *
from models.raw_data_models import *
from models.build_2dmodel import build_2d_model


def test_VggIMG():
    # Setup
    inputs = torch.rand(2, 1, 9, 51)  # (N, C, H, W)
    model_1 = build_2d_model('VggIMG', 'mesa', 3, 50)
    # Exercise
    output_1 = model_1(inputs)
    # Verify
    assert list(output_1.shape) == [2, 3]


def test_VggIMGSum():
    # Setup
    inputs = torch.rand(2, 1, 9, 51)  # (N, C, H, W)
    model_1 = build_2d_model('VggIMGSum', 'mesa', 3, 50)
    # Exercise
    output_1 = model_1(inputs)
    # Verify
    assert list(output_1.shape) == [2, 3]


def test_VggIMGRes():
    # Setup
    inputs = torch.rand(2, 1, 9, 51)  # (N, C, H, W)
    model_1 = build_2d_model('VggIMGRes', 'mesa', 3, 50)
    # Exercise
    output_1 = model_1(inputs)
    # Verify
    assert list(output_1.shape) == [2, 3]


def test_VggIMGResSum():
    # Setup
    inputs = torch.rand(2, 1, 9, 51)  # (N, C, H, W)
    model_1 = build_2d_model('VggIMGResSum', 'mesa', 3, 50)
    # Exercise
    output_1 = model_1(inputs)
    # Verify
    assert list(output_1.shape) == [2, 3]


def test_VggAcc79F174_7_RM():
    inputs = torch.rand(2, 9, 101)
    model_1 = build_model('VggAcc79F174_7_RM', 'mesa', 3, 100, modality='none')
    output_1 = model_1(inputs)
    # Verify
    assert list(output_1.shape) == [2, 3]


def test_VggAcc79F174_7_RM_Raw_Appl_1():
    # Setup
    # (N, C, H, W)
    acc_input = torch.rand(2, 3, 3030)
    hrv_input = torch.rand(2, 6, 101)
    model_1 = build_raw_model('VggAcc79F174_7_RM_Raw_Appl_1', 'apple_raw', seq_len=100, num_classes=3)
    # Exercise
    output_1 = model_1(acc_input, hrv_input)
    # Verify
    assert list(output_1.shape) == [2, 3]


# def test_VggAcc79F174_7_RM_RAW_APPL():
#     acc_input = torch.rand(2, 3, 3030)
#     hrv_input = torch.rand(2, 6, 101)
#     model_1 = build_raw_model('VggAcc79F174_7_RM_Raw_Appl', 'apple_raw', seq_len=100, num_classes=3)
#     output_1 = model_1(acc_input, hrv_input)
#     assert output_1.shape == (2,3)
#
#
# def test_VggAcc79F174_7_RM_RAW_APPL1():
#     acc_input = torch.rand(2, 3, 3030)
#     hrv_input = torch.rand(2, 6, 101)
#     model_1 = build_raw_model('VggAcc79F174_7_RM_Raw_Appl_1', 'apple_raw', seq_len=100, num_classes=3)
#     output_1 = model_1(acc_input, hrv_input)
#     assert output_1.shape == (2,3)


def test_ResPlus_Raw_Appl_1():
    acc_input = torch.rand(2, 3, 3030)
    hrv_input = torch.rand(2, 6, 101)
    model_1 = build_raw_model('ResPlus_Raw_Appl_1', 'apple_raw', seq_len=100, num_classes=3)
    output_1 = model_1(acc_input, hrv_input)
    assert output_1.shape == (2, 3)


# def test_ResPlus_Raw_Appl():
#     acc_input = torch.rand(2, 3, 3030)
#     hrv_input = torch.rand(2, 6, 101)
#     model_1 = build_raw_model('ResPlus_Raw_Appl', 'apple_raw', seq_len=100, num_classes=3)
#     output_1 = model_1(acc_input, hrv_input)
#     assert output_1.shape == (2, 3)


def test_RawAccFeatureExtraction():
    acc_input = torch.rand(2, 3, 3030)
    model_1 = RawAccFeatureExtraction(raw_acc_in_ch=3)
    output_1 = model_1(acc_input)
    assert output_1.shape == (2, 512, 25)


def test_VggRawSplitModal():
    acc_input = torch.rand(2, 3, 3030)
    hrv_input = torch.rand(2, 6, 101)
    model_1 = build_raw_model('VggRawSplitModal', 'apple_raw', seq_len=100, num_classes=3)
    output_1 = model_1(acc_input, hrv_input)
    assert output_1[1].shape == (2, 3)


def test_VggRawSplitModal():
    acc_input = torch.rand(2, 3, 3030)
    hrv_input = torch.rand(2, 6, 101)
    model_1 = build_raw_model('VggRawSplitModalAdd', 'apple_raw', seq_len=100, num_classes=3)
    output_1 = model_1(acc_input, hrv_input)
    assert output_1[1].shape == (2, 3)


def test_ResPlusRawSplitModalCon():
    acc_input = torch.rand(2, 3, 3030)
    hrv_input = torch.rand(2, 6, 101)
    model_1 = build_raw_model('ResPlusRawSplitModalCon', 'apple_raw', seq_len=100, num_classes=3,
                              modality='act')
    output_1 = model_1(acc_input, hrv_input)
    assert output_1[1].shape == (2, 3)


def test_ResPlusRawSplitModalPlus():
    acc_input = torch.rand(2, 3, 3030)
    hrv_input = torch.rand(2, 6, 101)
    model_1 = build_raw_model('ResPlusRawSplitModalPlus', 'apple_raw', seq_len=100, num_classes=3,
                              modality='none')
    output_1 = model_1(acc_input, hrv_input)
    assert output_1[1].shape == (2, 3)


def test_ResRawPlusSplitModal_SANTiDimMatAttMod1NLayer1Con():
    acc_input = torch.rand(2, 3, 3030)
    hrv_input = torch.rand(2, 6, 101)
    model_1 = build_raw_model('ResRawPlusSplitModal_SANTiDimMatAttMod1NLayer1Con', 'apple_raw', seq_len=100, num_classes=3,
                              modality='act')
    output_1 = model_1(acc_input, hrv_input)
    assert output_1[1].shape == (2, 3)


def test_ResPlusRawSplitModal_BiLinear():
    acc_input = torch.rand(2, 3, 3030)
    hrv_input = torch.rand(2, 6, 101)
    model_1 = build_raw_model('ResPlusRawSplitModal_BiLinear', 'apple_raw', seq_len=100, num_classes=3,
                              modality='none')
    output_1 = model_1(acc_input, hrv_input)
    assert output_1[1].shape == (2, 3)


def test_VggRaw_BiLinear():
    acc_input = torch.rand(2, 3, 3030)
    hrv_input = torch.rand(2, 6, 101)
    model_1 = build_raw_model('VggRaw_BiLinear', 'apple_raw', seq_len=100, num_classes=3,
                              modality='none')
    output_1 = model_1(acc_input, hrv_input)
    assert output_1[1].shape == (2, 3)


def test_VggRaw2DConcate():
    acc_input = torch.rand(2, 3, 3030)
    hrv_input = torch.rand(2, 6, 101)
    model_1 = build_raw_model('VggRaw2DConcate', 'apple_raw', seq_len=100, num_classes=3,
                              modality='none')
    output_1 = model_1(acc_input, hrv_input)
    summary(model_1, [(3, 3030), (6, 101)], device='cpu')

    assert output_1[1].shape == (2, 3)

def test_VggRaw2DSum():
    acc_input = torch.rand(2, 3, 3030)
    hrv_input = torch.rand(2, 6, 101)
    model_1 = build_raw_model('VggRaw2DSum', 'apple_raw', seq_len=100, num_classes=3,
                              modality='none')
    summary(model_1, [(3, 3030), (6, 101)], device='cpu')
    output_1 = model_1(acc_input, hrv_input)
    assert output_1[1].shape == (2, 3)


def test_VggRaw2DResConcate():
    acc_input = torch.rand(2, 3, 3030)
    hrv_input = torch.rand(2, 6, 101)
    model_1 = build_raw_model('VggRaw2DResConcate', 'apple_raw', seq_len=100, num_classes=3,
                              modality='none')
    summary(model_1, [(3, 3030), (6, 101)], device='cpu')
    output_1 = model_1(acc_input, hrv_input)
    assert output_1[1].shape == (2, 3)

def test_VggRaw2DResSum():
    acc_input = torch.rand(2, 3, 3030)
    hrv_input = torch.rand(2, 6, 101)
    model_1 = build_raw_model('VggRaw2DResSum', 'apple_raw', seq_len=100, num_classes=3,
                              modality='none')
    summary(model_1, [(3, 3030), (6, 101)], device='cpu')
    output_1 = model_1(acc_input, hrv_input)
    assert output_1[1].shape == (2, 3)