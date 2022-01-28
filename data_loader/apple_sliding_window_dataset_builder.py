from sklearn.preprocessing import OneHotEncoder
import h5py
from utilities.utils import *
from sklearn.model_selection import train_test_split
from sleep_stage_config import *


class AppleSleepDataBuilder(object):
    """
    Apple Watch Dataset Builder
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
        # self.ml_feature_list = []
        self.__prepare_feature_list__()

    def __prepare_feature_list__(self):
        if self.modality == "all":
            self.dl_feature_list = ["activity", "min_hr", "max_hr", "mean_hr", "skw_hr", "kurt_hr", "std_hr"]
        elif self.modality == "hr":
            self.dl_feature_list = ["min_hr", "max_hr", "mean_hr", "skw_hr", "kurt_hr", "std_hr"]
        elif self.modality == "acc":
            self.dl_feature_list = ["activity"]
        elif self.modality == "hr":
            self.dl_feature_list = ["mean_hr"]

    @staticmethod
    def __check_seq_len__(seq_len):
        if seq_len not in [100, 50, 20]:
            raise Exception("seq_len i error")

    def build_windowed_cache_data(self, win_len):
        """
        Build the sliding window data and save it for experiments.
        @param win_len: window length to build 20, 50, 100
        """
        self.__check_seq_len__(win_len)
        assert self.modality == "all", "building up cache only works when modality is 'all', " \
                                       "as other modalities included in the cache data file"
        print("Loading H5 dataset....")
        df_train, df_test, feat_names = load_h5_df_train_test_dataset(self.config.APPLE_HRV30_ACC_STD_PATH)
        cache_path = self.config.APPLE_NN_ACC_HRV % win_len
        print("building cached dataset for window length: %s ....." % win_len)
        x_train, y_train = get_data(df_train, win_len, self.dl_feature_list, pid_col_name="appleid",
                                    gt_col_name="stages")
        x_test, y_test = get_data(df_test, win_len, self.dl_feature_list, pid_col_name="appleid", gt_col_name="stages")
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, random_state=42,
                                                          shuffle=False)
        with h5py.File(cache_path, 'w') as data:
            data["x_train"] = x_train
            data["y_train"] = y_train
            data["x_val"] = x_val
            data["y_val"] = y_val
            data["x_test"] = x_test
            data["y_test"] = y_test
            data.close()

    def _load_std_df(self):
        """
        LOOCV will concatenate train and test into a single frame. The train test index for each fold in CV
        is stored in a separate file
        """
        df_train, df_test, feature_name = load_h5_df_train_test_dataset(self.config.APPLE_HRV30_ACC_STD_PATH)
        df_train['stages'] = df_train['stages'].apply(lambda x: cast_sleep_stages(x, classes=self.num_classes))
        df_test['stages'] = df_test['stages'].apply(lambda x: cast_sleep_stages(x, classes=self.num_classes))
        df = pd.concat([df_train, df_test])
        return df, feature_name


class AppleSleepLOOCVDataBuilder(AppleSleepDataBuilder):
    """
    this class will build the dataset in one file but not split it,
    we leave that job to dataloader
    """

    def __init__(self, *args, **kwargs):
        super(AppleSleepLOOCVDataBuilder, self).__init__(*args, **kwargs)

    def build_windowed_cache_data(self, win_len: int):
        df, feature_list = self._load_std_df()
        df = df.rename(columns={"appleid": "pid", "linetime": "line"})
        pid_list = df.pid.unique().tolist()
        sorter_index = dict(zip(pid_list, range(1, len(pid_list) + 1)))
        df['master_idx'] = df['pid'].map(sorter_index)
        df['master_idx'] = df['master_idx'] * 10000000.0 + df['line']
        df.sort_values(['master_idx'], ascending=[True], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['window_idx'] = df.index
        x, y = get_data(df, win_len, self.dl_feature_list, pid_col_name="pid", gt_col_name="stages")
        assert np.abs(df['activity'] - x[:, int(win_len / 2), 0]).sum() == 0, \
            print("build dataset error, misalignment on df index")
        with h5py.File(self.config.APPLE_LOOCV_ALL_WINDOWED % win_len, 'w') as data:
            data["df_values"] = df.values
            data['x'] = x
            data['y'] = y
            data['columns'] = df.columns.values.astype("S")
            data.close()
            print('h5 dataset is saved to %s' % self.config.APPLE_LOOCV_ALL_WINDOWED)
        print("all done")


if __name__ == '__main__':
    config = Config()
    window_len = 100
    apple_windowed_builder = AppleSleepLOOCVDataBuilder(cfg=config, modality="all", num_classes=3, seq_len=window_len)
    apple_windowed_builder.build_windowed_cache_data(window_len)
