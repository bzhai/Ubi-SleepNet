from utilities.utils import build_windowed_data
from utilities.utils import load_h5_df_train_test_dataset, get_data, cast_sleep_stages
from sleep_stage_config import *
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class WindowedFrameAppleRAWDataLoader(torch.utils.data.Dataset):
    def __init__(self, acc_data, hrv_data, target, idx, transform=None):
        self.acc_data = torch.from_numpy(acc_data).float()
        self.acc_data = self.acc_data.permute(0, 2, 1)
        self.hrv_data = torch.from_numpy(hrv_data).float()
        self.hrv_data = self.hrv_data.permute(0, 2, 1)  #  set it to batch_num, channel, time_dim
        self.idx = torch.from_numpy(idx)
        self.target = torch.from_numpy(target).long()
        self.transform = transform

    def __getitem__(self, index):
        hrv_x = self.hrv_data[index]
        acc_x = self.acc_data[index]
        y = self.target[index]
        i = self.idx[index]
        return acc_x, hrv_x, y, i

    def __len__(self):
        return len(self.target)


class WindowedFrameMESARAWDataLoader(torch.utils.data.Dataset):
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


def get_raw_dataloader_by_id(pid, cfg: Config, shuffle, batch_size, data_set, seq_len, apple_acc_hz=1):
    import h5py as h5py
    if data_set == "apple_raw":
        pid_raw_acc_path = os.path.join(cfg.APPLE_CROPPED_RAW_PATH, f"{str(pid)}_cleaned_resampled_"
                                                                    f"{str(apple_acc_hz)}_hz.out")
        raw_acc = pd.read_csv(pid_raw_acc_path, delimiter=' ', header=None).values
        raw_acc = raw_acc[:raw_acc.shape[0]-30, 1:]
        outputs = []
        for i in np.arange(3):
            sig = raw_acc[:, i].reshape(-1, 30)  # e.g. 200 x 30
            out = build_windowed_data(sig=sig, sampling_rate=1, epoch_len=30, win_len=seq_len+1)
            assert out.shape == (sig.shape[0], 30*(seq_len+1))
            outputs.append(np.expand_dims(out, -1))
        raw_acc_x = np.concatenate(outputs, axis=-1)
        cache_path = cfg.APPLE_LOOCV_ALL_WINDOWED % seq_len
        with h5py.File(cache_path, 'r') as data:
            df_data = data["df_values"][:]
            x = data["x"][:]
            y = data["y"][:]
            columns = data["columns"][:].astype(str).tolist()
            data.close()
        df = pd.DataFrame(df_data, columns=columns)
        pid_idx = df[df.pid == pid]['window_idx'].values.astype(int)
        x_hrv = x[pid_idx, :, :][:, :, 1:]  # remove the activity counts only keep the hrv features
        y_stage = y[pid_idx]
        data_ds = WindowedFrameAppleRAWDataLoader(raw_acc_x, x_hrv, y_stage, pid_idx)
        data_loader = DataLoader(
            data_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
    return data_loader


def get_raw_test_df(pid, cfg: Config, dataset, num_classes, seq_len):
    import h5py as h5py
    if dataset == "apple_raw":
        with h5py.File(cfg.APPLE_LOOCV_ALL_WINDOWED % seq_len, 'r') as data:
            df_value = data["df_values"][:]
            df_columns = data['columns'][:].astype(str).tolist()
            data.close()
        df_test = pd.DataFrame(df_value, columns=df_columns)
        df_test = df_test[df_test['pid'] == pid].copy(deep=True)
    return df_test

