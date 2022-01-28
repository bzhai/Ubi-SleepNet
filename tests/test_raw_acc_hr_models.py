from models.build_raw_acc_hr_model import build_raw_acc_hr_model
import torch


def test_VggAcc79F174_7_RM_Raw_Appl_1_hr():
    # Setup
    acc_input = torch.rand(2, 3, 3030)
    hrv_input = torch.rand(2, 1, 101)
    model_1 = build_raw_acc_hr_model('VggAcc79F174_7_RM_Raw_Appl_1_hr', 'apple_raw', seq_len=100, num_classes=3)
    # Exercise
    output_1 = model_1(acc_input, hrv_input)
    # Verify
    assert list(output_1.shape) == [2, 3]


def test_ResPlus_Raw_Appl_1_hr():
    # Setup
    acc_input = torch.rand(2, 3, 3030)
    hrv_input = torch.rand(2, 1, 101)
    model_1 = build_raw_acc_hr_model('ResPlus_Raw_Appl_1_hr', 'apple_raw', seq_len=100, num_classes=3)
    # Exercise
    output_1 = model_1(acc_input, hrv_input)
    # Verify
    assert list(output_1.shape) == [2, 3]


def test_VggRawSplitModal_hr():
    """
    hybrid fusion for VGG concatenation
    @return:
    """
    # Setup
    acc_input = torch.rand(2, 3, 3030)
    hrv_input = torch.rand(2, 1, 101)
    model_1 = build_raw_acc_hr_model('VggRawSplitModal_hr', 'apple_raw', seq_len=100, num_classes=3)
    # Exercise
    output_1 = model_1(acc_input, hrv_input)
    # Verify
    assert list(output_1[1].shape) == [2, 3]


def test_VggRawSplitModalAdd_hr():
    """
    hybrid fusion for VGG addition
    @return:
    """
    # Setup
    acc_input = torch.rand(2, 3, 3030)
    hrv_input = torch.rand(2, 1, 101)
    model_1 = build_raw_acc_hr_model('VggRawSplitModalAdd_hr', 'apple_raw', seq_len=100, num_classes=3)
    # Exercise
    output_1 = model_1(acc_input, hrv_input)
    # Verify
    assert list(output_1[1].shape) == [2, 3]


def test_VggRawSANTiDimMatAttMod1NLayer1Con_hr():
    """
    Hybrid fusion for attention using VGG
    @return:
    """
    # Setup
    acc_input = torch.rand(2, 3, 3030)
    hrv_input = torch.rand(2, 1, 101)
    model_1 = build_raw_acc_hr_model('VggRawSANTiDimMatAttMod1NLayer1Con_hr', 'apple_raw', seq_len=100, num_classes=3,
                                     modality='act')
    # Exercise
    output_1 = model_1(acc_input, hrv_input)
    # Verify
    assert list(output_1[1].shape) == [2, 3]


def test_ResPlusRawSplitModalCon_hr():
    """
    Hybrid fusion for concatenation using ResVGG
    @return:
    """
    # Setup
    acc_input = torch.rand(2, 3, 3030)
    hrv_input = torch.rand(2, 1, 101)
    model_1 = build_raw_acc_hr_model('ResPlusRawSplitModalCon_hr', 'apple_raw', seq_len=100, num_classes=3,
                                     )
    # Exercise
    output_1 = model_1(acc_input, hrv_input)
    # Verify
    assert list(output_1[1].shape) == [2, 3]


def test_ResPlusRawSplitModalPlus_hr():
    """
    Hybrid fusion for concatenation using ResVGG
    @return:
    """
    # Setup
    acc_input = torch.rand(2, 3, 3030)
    hrv_input = torch.rand(2, 1, 101)
    model_1 = build_raw_acc_hr_model('ResPlusRawSplitModalPlus_hr', 'apple_raw', seq_len=100, num_classes=3,
                                     )
    # Exercise
    output_1 = model_1(acc_input, hrv_input)
    # Verify
    assert list(output_1[1].shape) == [2, 3]


def test_ResRawPlusSplitModal_SANTiDimMatAttMod1NLayer1Con_hr():
    """
    ybrid fusion for concatenation using ResVGG
    @return:
    """
    # Setup
    acc_input = torch.rand(2, 3, 3030)
    hrv_input = torch.rand(2, 1, 101)
    model_1 = build_raw_acc_hr_model('ResRawPlusSplitModal_SANTiDimMatAttMod1NLayer1Con_hr', 'apple_raw', seq_len=100, num_classes=3,
                                     modality='act')
    # Exercise
    output_1 = model_1(acc_input, hrv_input)
    # Verify
    assert list(output_1[1].shape) == [2, 3]


def test_VggRaw_BiLinear_hr():
    """
    Hybrid fusion for concatenation using ResVGG
    @return:
    """
    # Setup
    acc_input = torch.rand(2, 3, 3030)
    hrv_input = torch.rand(2, 1, 101)
    model_1 = build_raw_acc_hr_model('VggRaw_BiLinear_hr', 'apple_raw', seq_len=100, num_classes=3,
                                     )
    # Exercise
    output_1 = model_1(acc_input, hrv_input)
    # Verify
    assert list(output_1[1].shape) == [2, 3]


def test_ResPlusRawSplitModal_BiLinear_hr():
    """
    Hybrid fusion for concatenation using ResVGG
    @return:
    """
    # Setup
    acc_input = torch.rand(2, 3, 3030)
    hrv_input = torch.rand(2, 1, 101)
    model_1 = build_raw_acc_hr_model('ResPlusRawSplitModal_BiLinear_hr', 'apple_raw', seq_len=100, num_classes=3,
                                     )
    # Exercise
    output_1 = model_1(acc_input, hrv_input)
    # Verify
    assert list(output_1[1].shape) == [2, 3]
