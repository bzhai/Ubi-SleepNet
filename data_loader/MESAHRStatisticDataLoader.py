
from sklearn.preprocessing import OneHotEncoder
import h5py
from torch.utils.data import DataLoader
import torch
from data_loader.TorchFrameDataLoader import WindowedFrameDataLoader
from utilities.utils import *
from sklearn.model_selection import train_test_split
from sleep_stage_config import *


class MESAAccHRStatisticDataLoader(object):
    """
    a dataset loader for actigraphy
    """

    def __init__(self, cfg, modality, num_classes, seq_len):
        self.config = cfg
        self.modality = modality
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.x_train = []
        self.y_train = []
        self.x_val = []
        self.y_val = []
        self.x_test = []
        self.y_test = []
        self.dl_feature_list = []
        self.__prepare_feature_list__()

    def __prepare_feature_list__(self):
        self.dl_feature_list = self.config.MESA_ACC_HR_STATISTIC_FEATURE_LIST

    @staticmethod
    def __check_seq_len__(seq_len):
        if seq_len not in [100, 50, 20]:
            raise Exception("seq_len i error")


    @staticmethod
    def cast_sleep_stages_and_onehot_encode(dataset, num_classes):
        if len(dataset.shape) < 2 and len(set(dataset)) != num_classes:
            dataset = cast_sleep_stages(dataset.astype(int), num_classes)
            if len(dataset.shape) < 2:
                dataset = np.expand_dims(dataset, -1)
            enc = OneHotEncoder(handle_unknown='ignore')
            dataset = enc.fit_transform(dataset).toarray()
            # dataset = tf.keras.utils.to_categorical(dataset, num_classes)
            return dataset
        else:
            return dataset

    def load_windowed_data(self):
        # h5_file = self.config.HRV30_ACC_STD_PATH
        print("Loading h5 dataset from %s" % self.config.MESA_ACC_HR_STATISTICS_STD_DATA_PATH)
        # _, dftest, featnames = load_h5_df_dataset(h5_file, useCache=True)
        print("The h5 dataset is loaded from %s" % self.config.MESA_ACC_HR_STATISTICS_STD_DATA_PATH)
        cache_path = self.config.MESA_NN_ACC_HR_STATISTIC % self.seq_len
        print("...Loading windowed cache dataset from %s" % cache_path)
        with h5py.File(cache_path, 'r') as data:
            if self.modality == "all":
                self.x_train = data["x_train"][:]
                self.y_train = data["y_train"][:]
                self.x_val = data["x_val"][:]
                self.y_val = data["y_val"][:]
                self.x_test = data["x_test"][:]
                self.y_test = data["y_test"][:]
            elif self.modality == "hrv":
                self.x_train = data["x_train"][:, :, 1:]
                self.y_train = data["y_train"][:]
                self.x_val = data["x_val"][:, :, 1:]
                self.y_val = data["y_val"][:]
                self.x_test = data["x_test"][:, :, 1:]
                self.y_test = data["y_test"][:]
            elif self.modality == "acc":
                self.x_train = np.expand_dims(data["x_train"][:, :, 0], -1)
                self.y_train = data["y_train"][:]
                self.x_val = np.expand_dims(data["x_val"][:, :, 0], -1)
                self.y_val = data["y_val"][:]
                self.x_test = np.expand_dims(data["x_test"][:, :, 0], -1)
                self.y_test = data["y_test"][:]
            elif self.modality == "hr":
                self.x_train = np.expand_dims(data["x_train"][:, :, 1], -1)
                self.y_train = data["y_train"][:]
                self.x_val = np.expand_dims(data["x_val"][:, :, 1], -1)
                self.y_val = data["y_val"][:]
                self.x_test = np.expand_dims(data["x_test"][:, :, 1], -1)
                self.y_test = data["y_test"][:]
            data.close()
        if len(self.y_train.shape) < 2 or len(set(self.y_train)) != self.num_classes:
            self.y_train = self.cast_sleep_stages_and_onehot_encode(self.y_train, self.num_classes)
        if len(self.y_test.shape) < 2 or len(set(self.y_test)) != self.num_classes:
            self.y_test = self.cast_sleep_stages_and_onehot_encode(self.y_test, self.num_classes)
        if len(self.y_val.shape) < 2 or len(set(self.y_val)) != self.num_classes:
            self.y_val = self.cast_sleep_stages_and_onehot_encode(self.y_val, self.num_classes)
        return self.y_train, self.y_test, self.y_val

    def load_df_dataset(self):
        df_train, df_test, feature_name = load_h5_dataset(self.config.MESA_ACC_HR_STATISTICS_STD_DATA_PATH)
        df_train['stages'] = df_train['stages'].apply(lambda x: cast_sleep_stages(x, classes=self.num_classes))
        df_test['stages'] = df_test['stages'].apply(lambda x: cast_sleep_stages(x, classes=self.num_classes))
        return df_train, df_test, feature_name

    def build_windowed_cache_data(self, win_len):
        self.__check_seq_len__(win_len)
        assert self.modality == "all", "building up cache only works when modality is 'all', " \
                                       "as other modalities included in the cache data file"
        print("Loading H5 dataset....")
        df_train, df_test, feat_names = load_h5_df_train_test_dataset(self.config.MESA_ACC_HR_STATISTICS_STD_DATA_PATH)
        cache_path = self.config.MESA_NN_ACC_HR_STATISTIC % win_len
        print("building cached dataset for window length: %s ....." % win_len)
        x_train, y_train = get_data(df_train, win_len, self.dl_feature_list)
        x_test, y_test = get_data(df_test, win_len, self.dl_feature_list)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, random_state=42, shuffle=False)
        with h5py.File(cache_path, 'w') as data:
            data["x_train"] = x_train
            data["y_train"] = y_train
            data["x_val"] = x_val
            data["y_val"] = y_val
            data["x_test"] = x_test
            data["y_test"] = y_test
            data.close()

    def build_windowed_cache_data_single_pid(self, win_len, num_classes):
        self.__check_seq_len__(win_len)
        assert self.modality == "all", "building up cache only works when modality is 'all', " \
                                       "as other modalities included in the cache data file"
        pid_df = pd.read_csv(r"../assets/mesa_16_fold.csv")
        print("Loading H5 dataset....")
        df_train, df_test, feat_names = load_h5_df_train_test_dataset(self.config.MESA_ACC_HR_STATISTICS_STD_DATA_PATH)
        df_all = pd.concat([df_train,df_test])
        df_all = df_all[['activity', 'min_hr', 'max_hr', 'mean_hr', 'skw_hr', 'kurt_hr', 'std_hr', 'line', 'stages', 'gt_sleep_block', 'mesaid']]
        cache_path = self.config.MESA_HRS_LOOCV
        pid_list = pid_df.pid.unique().tolist()
        for pid in pid_list:
            print("building cached dataset for window length: %s ....." % win_len)
            tmp_train = df_all[df_all['mesaid'] == pid].copy(deep=True)
            x, y = get_data(tmp_train, win_len, self.dl_feature_list)
            y = cast_sleep_stages(y,num_classes)
            with h5py.File(os.path.join(cache_path, r"%04d.h5" % int(pid)), 'w') as data:
                data["x"] = x
                data["y"] = y
                data["df_values"] = tmp_train.values.astype(np.float64)
                data["columns"] = tmp_train.columns.values.astype("S")
                data.close()

    def load_ml_data(self):
        df_train, df_test, _ = self.load_df_dataset()
        self.x_train = df_train[self.ml_feature_list]
        self.y_train = df_train["stages"]
        self.x_test = df_test[self.ml_feature_list]
        self.y_test = df_test["stages"]


def get_dataloader_by_id(pid, cfg: Config, shuffle, batch_size, seq_len):
    data_path = os.path.join(cfg.MESA_HRS_LOOCV, r"%04d.h5" % pid)
    with h5py.File(data_path, 'r') as data:
        df_data = data["df_values"][:]
        x = data["x"][:]
        y = data["y"][:]
        columns = data["columns"][:].astype(str).tolist()
        data.close()
    df = pd.DataFrame(df_data, columns=columns)
    idx = df.line.values
    ds = WindowedFrameDataLoader(x, y, idx)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    return loader


def get_test_df_by_id(pid, cfg: Config, dataset, num_classes):
    if dataset == "mesa_loocv":
        data_path = os.path.join(cfg.MESA_HRS_LOOCV, r"%04d.h5" % int(pid))
        with h5py.File(data_path, 'r') as data:
            df_data = data["df_values"][:]
            columns = data["columns"][:].astype(str).tolist()
            data.close()
        df_test = pd.DataFrame(df_data, columns=columns)
    df_test['stages'] = df_test['stages'].apply(lambda x: cast_sleep_stages(x, classes=num_classes))
    df_test.rename(columns={'mesaid': 'pid'}, inplace=True)
    return df_test


if __name__ == '__main__':
    config = Config()
    builder = MESAAccHRStatisticDataLoader(config, "all", 3, 20)

    builder.build_windowed_cache_data_single_pid(100, 3)
