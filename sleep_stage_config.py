import os
import platform


class Config(object):
    def __init__(self):
        # {"activity": "Activity Counts",
        #  "mean_nni": "Mean NNI",
        #  "sdnn": "Standard Deviation of NN Intervals",
        #  "sdsd": "Successive RR Interval Differences",
        #  "vlf": "Very-Low-Frequency Band",
        #  "lf": "Low-Frequency Band",
        #  "hf": "High-Frequency Band",
        #  "lf_hf_ratio": "A ratio of Low Frequency to High Frequency",
        #  "total_power": "The Signal's Power Intensity"
        #  }
        self.MESA_ACC_HR_STATISTIC_FEATURE_LIST = ["activity", "min_hr", "max_hr", "mean_hr", "skw_hr", "kurt_hr",
                                                   "std_hr"]
        self.SUMMARY_FILE_PATH = r"./exp_results.csv"
        self.MESA_LOOCV_PID_PATH = r"./assets/mesa_16_fold.csv"
        if platform.uname()[1] == 'BB-WIN8':
            self.HR_PATH = "Dataset/MESA/annotations-rpoints"
            self.ACC_PATH = "/Dataset/MESA/actigraphy"
            self.OVERLAP_PATH = "/Dataset/MESA/mesa-actigraphy-psg-overlap.csv"
            self.HRV30_ACC_STD_PATH = "/Dataset/MESA/HRV30s_ACC30s_H5/std/hrv30s_acc30s_full_feat_stand.h5"
            self.NN_ACC_HRV_STD = "/Dataset/MESA/HRV30s_ACC30s_H5/std/nn_acc_hrv30s_%d.h5"
            self.MESA_ACC_HR_STATISTICS_STD_DATA_PATH = "/Dataset/MESA/HR_STATISTIC30s_ACC30s_H5/hr_statistic_30s_acc30s_full_feat_stand.h5"
            self.MESA_NN_ACC_HR_STATISTIC = "/Dataset/MESA/HR_STATISTIC30s_ACC30s_H5/nn_acc_hr_statistic_30s_%d.h5"
            self.MESA_ACC_HR_STATISTIC_CSV_ALIGNED = "/Dataset/MESA/Aligned_ACC_HR_STATISTIC_CSV"
            self.MESA_ACC_HR_STATISTIC_FEATURE = "/Dataset/MESA/mesa_acc_hr_statistic_feature_list.csv"
            # apple data set settings
            self.APPLE_NN_ACC_HRV = r"/Dataset/Apple_watch_sleep_dataset/outputs/features/apple_nn_acc_hrv30s_%d.h5"
            self.APPLE_LOOCV_ALL_WINDOWED = r"/Dataset/Apple_watch_sleep_dataset/outputs/features/apple_loocv_windowed_%d.h5"
            self.APPLE_HRV30_ACC_STD_PATH = r"/Dataset/Apple_watch_sleep_dataset/outputs/features/apple_hr30s_acc30s_full_feat_stand.h5"
            self.APPLE_LOOCV_PID_PATH = r"/Dataset/Apple_watch_sleep_dataset/outputs/features/apple_16_fold.csv"
            self.APPLE_CROPPED_RAW_PATH = r"/Dataset/Apple_watch_sleep_dataset/outputs/cropped"

            self.TRAIN_TEST_SPLIT = "/assets/train_test_pid_split.csv"
            self.SUMMARY_FOLDER_DICT = {"s": r"/sp_exp_results.csv",
                                        "r": r"/rp_exp_results.csv", }
            self.MESA_HRS_LOOCV = r"/Dataset/MESA/mesa_hrs_loocv"
