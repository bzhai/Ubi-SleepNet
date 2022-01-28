import pandas as pd
from pathlib import Path
import numpy as np
from utilities.utils import make_one_block, standardize_df_given_feature
import os
import pickle
from tqdm import tqdm


def get_project_root() -> Path:
    return Path(r'\Dataset\Apple_watch_sleep_dataset')


FEATURE_FILE_PATH = r"\Dataset\features"
OUTPUT_PATH_FILE = os.path.join(FEATURE_FILE_PATH, "apple_hr30s_acc30s_full_feat_stand.h5")
subject_ids = [46343, 3509524, 5132496, 1066528, 5498603, 2638030, 2598705, 5383425, 1455390, 4018081, 9961348,
                    1449548, 8258170, 781756, 9106476, 8686948, 8530312, 3997827, 4314139, 1818471, 4426783,
                    8173033, 7749105, 5797046, 759667, 8000685, 6220552, 844359, 9618981, 1360686,
                    8692923]


def fix_apple_sleep_stages(data, classes=5):
    if type(data) is np.ndarray:
        data[data == 4] = 3  # non-REM 3 combined NREM4 to NREM3
        data[data == 5] = 4  # index 5 move to index 4
        return data
    else:
        # this is for a scalar
        # dataset=0 wake, dataset=1:non-REM, dataset=2:non-REM, dataset=3:non-REM, dataset=4:REM
        stages_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5:4}
        return stages_dict[data]
        return data


def build_hold_out_h5df():
    tmp = []
    for subject_id in tqdm(subject_ids):
        # print("Processing pid: %s" % subject_id)
        # read hr features
        # read act features
        hr_features = pd.read_csv(str(get_project_root().joinpath('outputs/features/', str(subject_id) + "_hr_feature.out")), delimiter=' ', header=None).values
        act_features = pd.read_csv(str(get_project_root().joinpath('outputs/features/', str(subject_id) + "_count_feature.out")), delimiter=' ', header=None).values
        sleep_stages = pd.read_csv(str(get_project_root().joinpath('outputs/features/', str(subject_id) + "_psg_labels.out")), delimiter=' ', header=None).values
        combined_dataset = np.hstack([act_features, hr_features, sleep_stages])
        df_columns = ["activity", "min_hr", "max_hr", "mean_hr", "skw_hr", "kurt_hr", "std_hr", "linetime", "stages"]
        feature_list = ["activity", "min_hr", "max_hr", "mean_hr", "skw_hr", "kurt_hr", "std_hr"]
        assert combined_dataset.shape[1] == len(df_columns), print("num of columns does not match ")
        df_tmp = pd.DataFrame(combined_dataset)
        df_tmp.columns = df_columns
        df_tmp["stages"] = fix_apple_sleep_stages(df_tmp["stages"].values)
        gt_true = df_tmp[df_tmp["stages"] > 0]
        if gt_true.empty:
            print("Ignoring subject's file %s" % subject_id)
            continue
        start_block = df_tmp.index.get_loc(gt_true.index[0])
        end_block = df_tmp.index.get_loc(gt_true.index[-1])
        df_tmp["gt_sleep_block"] = make_one_block(df_tmp["stages"], start_block, end_block)
        df_tmp["appleid"] = subject_id
        tmp.append(df_tmp)
    test_proportion = 0.2
    whole_df = pd.concat(tmp)
    whole_df = whole_df.reset_index(drop=True)
    del tmp
    np.random.seed(9)
    np.random.shuffle(subject_ids)
    test_id_file = os.path.join(FEATURE_FILE_PATH, "apple_watch_test.txt")
    train_id_file = os.path.join(FEATURE_FILE_PATH, "apple_watch_train.txt")
    if os.path.exists(train_id_file) & os.path.exists(test_id_file):
        uid_train = np.loadtxt(train_id_file)
        uid_test = np.loadtxt(test_id_file)
    else:
        test_idx = int(len(subject_ids) * test_proportion)
        uid_test = np.asarray(subject_ids[:test_idx])
        uid_train = np.asarray(subject_ids[test_idx:])
        np.savetxt(test_id_file, uid_test, fmt='%d')
        np.savetxt(train_id_file, uid_train, fmt='%d')
    train_idx = whole_df[whole_df["appleid"].apply(lambda x: x in uid_train)].index
    dftrain = whole_df.iloc[train_idx].copy()
    test_idx = whole_df[whole_df["appleid"].apply(lambda x: x in uid_test)].index
    dftest = whole_df.iloc[test_idx].copy()
    print("start standardisation on df_train....")
    scaler = standardize_df_given_feature(dftrain, feature_list, df_name="dftrain", simple_method=True)
    print("start standardisation on df_test....")
    standardize_df_given_feature(dftest, feature_list, scaler, df_name="dftest", simple_method=True)
    store = pd.HDFStore(OUTPUT_PATH_FILE, 'w')
    store["train"] = dftrain
    store["test"] = dftest
    store["featnames"] = pd.Series(feature_list)
    store.close()
    print('h5 dataset is saved to %s' % OUTPUT_PATH_FILE)
    with open(os.path.join(OUTPUT_PATH_FILE + '_std_transformer'), "wb") as f:
        pickle.dump(scaler, f)
    print("all completed")


if __name__ == '__main__':
    build_hold_out_h5df()
