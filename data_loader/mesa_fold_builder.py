import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import csv
total_fold = 16
k_fold = KFold(n_splits=total_fold)
pids = pd.read_csv("../assets/train_test_pid_split.csv")
pids = pids.uids.tolist()
np.random.shuffle(pids)
pids = np.asarray(pids[:31])

fold_df = [] # pd.DataFrame(columns=["fold_num", "set_name", "pid"])
for fold_idx, (train_set, test_set) in enumerate(k_fold.split(pids)):
    np.random.shuffle(train_set)
    new_train_set = train_set[:int(np.floor(len(train_set)*0.8))]
    val_set = train_set[int(np.floor(len(train_set)*0.8)):]
    tmp_df_list = []
    for set_type, pid_set in {"train": new_train_set, "val": val_set, "test": test_set}.items():
        tmp_df = pd.DataFrame({"fold_num": [fold_idx]*len(pid_set),
                               "set_type": [set_type]*len(pid_set),
                               "pid": pids[pid_set]})
        tmp_df_list.append(tmp_df)
    tmp_df_list = pd.concat(tmp_df_list, axis=0, ignore_index=True)
    fold_df.append(tmp_df_list)
fold_df = pd.concat(fold_df, axis=0, ignore_index=True)
fold_df.to_csv(r"../assets/mesa_%s_fold.csv" % total_fold,
               index=False)
