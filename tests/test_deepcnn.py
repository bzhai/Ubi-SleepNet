from models.build_model import build_model
from models.build_2dmodel import *
from models.bilinear_model import BilinearFusion


def test_bilinear():
    image = torch.rand(10, 512, 12)
    question = torch.rand(10, 512, 12)
    model = BilinearFusion(512, 1024)
    output = model(image, question)
    assert output.shape == (10, 1024)
    print("code can be run")


def test_VggAcc79F174ResdPlus():
    """
    This is the test case for early stage fusion ResDeepCNN
    """
    x = torch.rand(2, 7, 51)
    model_1 = build_model('VggAcc79F174ResdPlus', 'apple', 3, 50, 'None')
    results = model_1(x)
    assert list(results[1].shape) == [2, 3]


def test_ResPlusSplitModalCon():
    """
    This is the test case for hybrid fusion ResDeepCNN concatenation method
    """
    x = torch.rand(2, 7, 51)
    model_1 = build_model('ResPlusSplitModalCon', 'apple', 3, 50, 'None')
    results = model_1(x)
    assert list(results[1].shape) == [2, 3]


def test_ResPlusSplitModalPlus():
    """
    This is the test case for hybrid fusion ResDeepCNN addition method
    """
    x = torch.rand(2, 7, 51)
    model_1 = build_model('ResPlusSplitModalPlus', 'apple', 3, 50, 'None')
    results = model_1(x)
    assert list(results[1].shape) == [2, 3]


def test_ResPlusSplitModal_BiLinear():
    """
    This is the test case for hybrid fusion ResDeepCNN bilinear method
    """
    x = torch.rand(2, 7, 51)
    model_1 = build_model('ResPlusSplitModal_BiLinear', 'apple', 3, 50, 'None')
    results = model_1(x)
    assert list(results[1].shape) == [2, 3]


def test_ResPlusSplitModal_SANTiDimMatAttMod1NLayer1Con():
    """
    This is the test case for hybrid fusion ResDeepCNN attention method
    """
    x = torch.rand(2, 7, 51)
    model_1 = build_model('ResPlusSplitModal_SANTiDimMatAttMod1NLayer1Con', 'apple', 3, 50, 'car')

    results = model_1(x)
    assert list(results[1].shape) == [2, 3]


def test_VggAcc79F174_RM_SANTiDimMatAttMod1NLayer1Con_20():
    """
    Test window length 21
    """
    # Setup
    x = torch.rand(2, 7, 21)
    model_1 = build_model('VggAcc79F174_RM_SANTiDimMatAttMod1NLayer1Con',
                          'apple', 3, 20, 'act'
                          )

    # Exercise
    results = model_1(x)
    # Verify
    assert list(results[1].shape) == [2, 3]


def test_ResPlusSplitModalCon_20():
    # Setup
    # set up 20 window length test examples
    x = torch.rand(2, 7, 21)
    model_1 = build_model('ResPlusSplitModalCon', 'apple', 3, 20, 'none')
    # Exercise
    results = model_1(x)
    # Verify
    assert list(results[1].shape) == [2, 3]


def test_ResPlusSplitModalCon_50():
    # Setup
    # set up 20 window length test examples
    x = torch.rand(2, 7, 51)
    # model_1 = ResPlusSplitModalCon(1, 6, 3, get_model_time_step_dim('ResPlusSplitModalCon', 50))
    model_1 = build_model('ResPlusSplitModalCon', 'apple', 3, 50, 'none')
    # Exercise
    results = model_1(x)
    # Verify
    assert list(results[1].shape) == [2, 3]


def test_ResPlusSplitModalPlus_50():
    # Setup
    # set up 20 window length test examples
    x = torch.rand(2, 7, 51)
    # model_1 = ResPlusSplitModalPlus(1, 6, 3, get_model_time_step_dim('ResPlusSplitModalCon', 50))
    model_1 = build_model('ResPlusSplitModalPlus', 'apple', 3, 50, 'none')
    # Exercise
    results = model_1(x)
    # Verify
    assert list(results[1].shape) == [2, 3]


def test_ResPlusSplitModal_SANTiDimMatAttMod1NLayer1Con_50():
    # Setup
    # set up 20 window length test examples
    x = torch.rand(2, 7, 51)
    model_1 = build_model('ResPlusSplitModal_SANTiDimMatAttMod1NLayer1Con', 'apple', 3, 50, 'car')
    # Exercise
    results = model_1(x)
    # Verify
    assert list(results[1].shape) == [2, 3]
