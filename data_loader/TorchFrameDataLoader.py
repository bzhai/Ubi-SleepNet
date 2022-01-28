"""
This script should only contain the frame to label data loaders
"""
import torch
from torch.utils.data import Dataset, DataLoader
from utilities.utils import *
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from sleep_stage_config import Config
import numpy as np
import pandas as pd


class WindowedFrameDataLoader2D(torch.utils.data.Dataset):
    def __init__(self, data, target, idx, transform=None):
        self.data = torch.from_numpy(data).float()
        self.data = self.data.permute(0, 2, 1)  #  set it to batch_num, channel, time_dim
        self.data = self.data.unsqueeze(1)
        self.idx = torch.from_numpy(idx)
        self.target = torch.from_numpy(target).long()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        i = self.idx[index]
        if self.transform:
            x = self.transform(x)
        return x, y, i

    def __len__(self):
        return len(self.data)


class WindowedFrameDataLoader(torch.utils.data.Dataset):
    def __init__(self, data, target, idx, transform=None):
        self.data = torch.from_numpy(data).float()
        self.data = self.data.permute(0, 2, 1)  #  set it to batch_num, channel, time_dim
        self.idx = torch.from_numpy(idx)
        self.target = torch.from_numpy(target).long()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        i = self.idx[index]
        if self.transform:
            x = self.transform(x)
        return x, y, i

    def __len__(self):
        return len(self.data)


def get_test_df(cfg:Config, dataset, num_classes, seq_len):
    if dataset == "apple":
        import h5py as h5py
        with h5py.File(cfg.APPLE_LOOCV_ALL_WINDOWED % seq_len, 'r') as data:
            df_value = data["df_values"][:]
            df_columns = data['columns'][:].astype(str).tolist()
            data.close()
        df_test = pd.DataFrame(df_value, columns=df_columns)
        # read fold split and sort them by apple id
        # fold_num_df = pd.read_csv(cfg.APPLE_CV_PID_PATH)
        # pid_ordered_list = fold_num_df[fold_num_df['set_type'] == "test"]["pid"].values.tolist()
        # df_test = df_test.rename(columns={"appleid": "pid", "linetime": "line"})
        # new_df = []
        # for pid in pid_ordered_list:
        #     new_df.append(df_test[df_test['pid']==pid])
        # new_df = pd.concat(new_df, axis=0, ignore_index=True)

    elif dataset == "mesa":
        df_train, df_test, feature_name = load_h5_df_train_test_dataset(cfg.HRV30_ACC_STD_PATH) # load_h5_df_dataset(cfg.NN_ACC_HRV % seq_len)
        df_test = df_test.rename(columns={"mesaid": "pid"})
        del df_train
    elif dataset == "mesa_hr_statistic":
        df_train, df_test, feature_name = load_h5_df_train_test_dataset(cfg.MESA_ACC_HR_STATISTICS_STD_DATA_PATH) # load_h5_df_dataset(cfg.NN_ACC_HRV % seq_len)
        df_test = df_test.rename(columns={"mesaid": "pid"})
        del df_train
    if len(df_test['stages'].unique()) != num_classes:
        df_test['stages'] = df_test['stages'].apply(lambda x: cast_sleep_stages(x, classes=num_classes))
    return df_test


def get_windowed_train_test_val_loader(cfg, batch_size, seq_len, num_classes, dataset, fold):
    """
    The method will read pre-windows acc and hrv data from H5PY
    """
    import h5py as h5py
    if dataset == "mesa":
        assert fold == 0, print("mesa dataset only has 1 fold")
        cache_path = cfg.NN_ACC_HRV_STD % seq_len
        with h5py.File(cache_path, 'r') as data:
            x_train = data["x_train"][:]
            y_train = data["y_train"][:]
            x_val = data["x_val"][:]
            y_val = data["y_val"][:]
            x_test = data["x_test"][:]
            y_test = data["y_test"][:]
            data.close()
        train_idx = np.arange(y_train.shape[0])
        val_idx = np.arange(x_val.shape[0])
        test_idx = np.arange(x_test.shape[0])
    elif dataset == "mesa_hr_statistic":
        assert fold == 0, print("mesa hr statistic dataset only has 1 fold")
        cache_path = cfg.MESA_NN_ACC_HR_STATISTIC % seq_len
        with h5py.File(cache_path, 'r') as data:
            x_train = data["x_train"][:]
            y_train = data["y_train"][:]
            x_val = data["x_val"][:]
            y_val = data["y_val"][:]
            x_test = data["x_test"][:]
            y_test = data["y_test"][:]
            data.close()
        train_idx = np.arange(y_train.shape[0])
        val_idx = np.arange(x_val.shape[0])
        test_idx = np.arange(x_test.shape[0])

    elif dataset == "apple":
        cache_path = cfg.APPLE_LOOCV_ALL_WINDOWED % seq_len
        with h5py.File(cache_path, 'r') as data:
            df_data = data["df_values"][:]
            x = data["x"][:]
            y = data["y"][:]
            columns = data["columns"][:].astype(str).tolist()
            data.close()
        df = pd.DataFrame(df_data, columns=columns)
        split_df = pd.read_csv(cfg.APPLE_LOOCV_PID_PATH)
        train_pid = split_df[(split_df['set_type']=="train") & (split_df['fold_num']==fold)]['pid'].values.tolist()
        val_pid = split_df[(split_df['set_type']=="val") & (split_df['fold_num']==fold)]['pid'].values.tolist()
        test_pid = split_df[(split_df['set_type']=="test") & (split_df['fold_num']==fold)]['pid'].values.tolist()

        train_idx = df[df.pid.isin(train_pid)]['window_idx'].values.astype(int)
        val_idx = df[df.pid.isin(val_pid)]['window_idx'].values.astype(int)
        test_idx = df[df.pid.isin(test_pid)]['window_idx'].values.astype(int)

        x_train = x[train_idx, :, :]
        x_val = x[val_idx, :, :]
        x_test = x[test_idx, :, :]
        y_train = y[train_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]
    else:
        raise ValueError('%s dataset is not found' % dataset)
    print("...Loading windowed cache dataset from %s" % cache_path)


    # make sure the sleep classes are casted if the not 5 stages
    if (len(y_train.shape) < 2) and (len(set(y_train))) != num_classes:
        y_train = cast_sleep_stages(y_train.astype(int), num_classes)
    if (len(y_test.shape) < 2) and (len(set(y_test))) != num_classes:
        y_test = cast_sleep_stages(y_test.astype(int), num_classes)
    if (len(y_val.shape) < 2) and (len(set(y_val))) != num_classes:
        y_val = cast_sleep_stages(y_val, num_classes)

    train_ds = WindowedFrameDataLoader(x_train, y_train, train_idx)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    test_ds = WindowedFrameDataLoader(x_test, y_test, test_idx)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    val_ds = WindowedFrameDataLoader(x_val, y_val, val_idx)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    return train_loader, test_loader, val_loader


def get_windowed_apple_loader(cfg, batch_size, num_classes):
    """
    The method will read pre-windows acc and hrv data from H5PY
    """
    cache_path = cfg.APPLE_LOOCV_ALL_WINDOWED
    import h5py as h5py
    with h5py.File(cache_path, 'r') as data:
        df_data = data["df_values"][:]
        x = data["x"][:]
        y = data["y"][:]
        columns = data["columns"][:].astype(str).tolist()
        data.close()
    df = pd.DataFrame(df_data, columns=columns)
    split_df = pd.read_csv(cfg.APPLE_LOOCV_PID_PATH)
    all_pid = split_df['pid'].values.tolist()

    all_idx = df[df.pid.isin(all_pid)]['window_idx'].values.astype(int)

    x = x[all_idx, :, :]
    y = y[all_idx]

    print("...Loading windowed cache dataset from %s" % cache_path)


    # make sure the sleep classes are casted if the not 5 stages
    if (len(y.shape) < 2) and (len(set(y))) != num_classes:
        y = cast_sleep_stages(y.astype(int), num_classes)

    ds = WindowedFrameDataLoader(x, y, all_idx)

    data_loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return data_loader


def get_windowed_train_test_val_loader_2D(cfg, batch_size, seq_len, num_classes, dataset, fold):
    """
    The method will read pre-windows acc and hrv data from H5PY
    """
    import h5py as h5py
    if dataset == "mesa":
        assert fold == 0, print("mesa dataset only has 1 fold")
        cache_path = cfg.NN_ACC_HRV_STD % seq_len
        with h5py.File(cache_path, 'r') as data:
            x_train = data["x_train"][:]
            y_train = data["y_train"][:]
            x_val = data["x_val"][:]
            y_val = data["y_val"][:]
            x_test = data["x_test"][:]
            y_test = data["y_test"][:]
            data.close()
        train_idx = np.arange(y_train.shape[0])
        val_idx = np.arange(x_val.shape[0])
        test_idx = np.arange(x_test.shape[0])
    elif dataset == "mesa_hr_statistic":
        assert fold == 0, print("mesa hr statistic dataset only has 1 fold")
        cache_path = cfg.MESA_NN_ACC_HR_STATISTIC % seq_len
        with h5py.File(cache_path, 'r') as data:
            x_train = data["x_train"][:]
            y_train = data["y_train"][:]
            x_val = data["x_val"][:]
            y_val = data["y_val"][:]
            x_test = data["x_test"][:]
            y_test = data["y_test"][:]
            data.close()
        train_idx = np.arange(y_train.shape[0])
        val_idx = np.arange(x_val.shape[0])
        test_idx = np.arange(x_test.shape[0])

    elif dataset == "apple":
        cache_path = cfg.APPLE_LOOCV_ALL_WINDOWED % seq_len
        with h5py.File(cache_path, 'r') as data:
            df_data = data["df_values"][:]
            x = data["x"][:]
            y = data["y"][:]
            columns = data["columns"][:].astype(str).tolist()
            data.close()
        df = pd.DataFrame(df_data, columns=columns)
        split_df = pd.read_csv(cfg.APPLE_LOOCV_PID_PATH)
        train_pid = split_df[(split_df['set_type']=="train") & (split_df['fold_num']==fold)]['pid'].values.tolist()
        val_pid = split_df[(split_df['set_type']=="val") & (split_df['fold_num']==fold)]['pid'].values.tolist()
        test_pid = split_df[(split_df['set_type']=="test") & (split_df['fold_num']==fold)]['pid'].values.tolist()

        train_idx = df[df.pid.isin(train_pid)]['window_idx'].values.astype(int)
        val_idx = df[df.pid.isin(val_pid)]['window_idx'].values.astype(int)
        test_idx = df[df.pid.isin(test_pid)]['window_idx'].values.astype(int)

        x_train = x[train_idx, :, :]
        x_val = x[val_idx, :, :]
        x_test = x[test_idx, :, :]
        y_train = y[train_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]
    else:
        raise ValueError('%s dataset is not found' % dataset)
    print("...Loading windowed cache dataset from %s" % cache_path)


    # make sure the sleep classes are casted if the not 5 stages
    if (len(y_train.shape) < 2) and (len(set(y_train))) != num_classes:
        y_train = cast_sleep_stages(y_train.astype(int), num_classes)
    if (len(y_test.shape) < 2) and (len(set(y_test))) != num_classes:
        y_test = cast_sleep_stages(y_test.astype(int), num_classes)
    if (len(y_val.shape) < 2) and (len(set(y_val))) != num_classes:
        y_val = cast_sleep_stages(y_val, num_classes)

    train_ds = WindowedFrameDataLoader2D(x_train, y_train, train_idx)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    test_ds = WindowedFrameDataLoader2D(x_test, y_test, test_idx)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    val_ds = WindowedFrameDataLoader2D(x_val, y_val, val_idx)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    return train_loader, test_loader, val_loader


def get_apple_loocv_ids(cfg, fold):
    split_df = pd.read_csv(cfg.APPLE_LOOCV_PID_PATH)
    train_pid = split_df[(split_df['set_type']=="train") & (split_df['fold_num']==fold)]['pid'].values.tolist()
    val_pid = split_df[(split_df['set_type']=="val") & (split_df['fold_num']==fold)]['pid'].values.tolist()
    test_pid = split_df[(split_df['set_type']=="test") & (split_df['fold_num']==fold)]['pid'].values.tolist()
    return train_pid, val_pid, test_pid

def get_mesa_loocv_ids(cfg:Config, fold):
    split_df = pd.read_csv(cfg.MESA_LOOCV_PID_PATH)
    train_pid = split_df[(split_df['set_type']=="train") & (split_df['fold_num']==fold)]['pid'].values.tolist()
    val_pid = split_df[(split_df['set_type']=="val") & (split_df['fold_num']==fold)]['pid'].values.tolist()
    test_pid = split_df[(split_df['set_type']=="test") & (split_df['fold_num']==fold)]['pid'].values.tolist()
    return train_pid, val_pid, test_pid

if __name__ == "__main__":
    cfg = Config()
    # train_loader, test_loader, val_loader = get_windowed_train_test_val_loader(cfg, 100, 100, 3, "apple")
    train_loader = get_windowed_apple_loader(cfg, 100, 3)
    for i, data in enumerate(train_loader):
        print(data[0].shape)
    print("output shape")