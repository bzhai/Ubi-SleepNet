from data_loader.TorchFrameDataLoader import get_apple_loocv_ids
from data_loader.raw_data_loader import *


def test_raw_apple_df_loader():
    cfg = Config()
    results = get_raw_test_df(3509524, cfg, 'apple_raw', 3, 100)
    assert results.shape[0] == 415


def test_get_raw_apple_dataset_by_id():
    cfg = Config()
    data_loader = get_raw_dataloader_by_id(46343, cfg, False, 100,'apple_raw', 100, 1)
    total_samples = 0
    for (acc, hrv, y, idx) in data_loader:
        total_samples += len(idx)
    print("total num of samples is:")
    assert True


def test_get_apple_loocv_ids():
    cfg = Config()
    for fold in np.arange(16):
        train_id, val_id, test_id = get_apple_loocv_ids(cfg, fold)
        if fold != 15:
            assert len(train_id) + len(val_id) == 15*2-1
        else:
            assert len(train_id) + len(val_id) == 15*2

def test_get_dis_dataloader():
    cfg = Config()
    tr, va, te = get_win_train_test_val_dis_loader(cfg, 64, 100, 3, "mesa", 0, 50, 'sleepage5c')
    assert tr.batch_size == 64


def test_get_dis_id_testdf():
    cfg = Config()
    df = get_dis_test_df(cfg, "mesa", 3)
    print(df.shape)
