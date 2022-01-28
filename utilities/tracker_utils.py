import errno
import os.path as osp
from zipfile import ZipFile
from datetime import datetime
import os
import time
import platform
from evaluation_metrics import *
import numpy as np
from pathlib import Path
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, \
    classification_report
from torch.utils.tensorboard import SummaryWriter
from utilities.utils import write_arguments_to_file, get_all_files_include_sub, sleep_class_name_mapping, \
    plot_save_confusion_matrix, generate_tsne, load_torch_model_param
import sys
from summarize_experiments import evaluate_sleep_alg, save_evaluation_summary

# ######################################################################################################################
# ############################## classes for tracking experiments start from here  #####################################
# ######################################################################################################################

class ClassificationTrackerBak(object):
    """
    final results:
    args columns, best eval epoch idx, best eval acc, best eval micro_f1, best eval macro_f1, best test acc,
    for each experiment we record:
        1. the args used for the experiment
        2. append record eval and test performance after each training epoch
        3. append the summary performance to a file.
        4. save the best evaluation model.
        5. copy all model files to tensorboard folder.
    """

    def __init__(self, args, tensorboard_dir, master_kpi_path, avg_method="macro"):
        self.args = args
        self.avg_method = avg_method
        # ******** setup the tensor board path
        self.time_of_run = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_root = tensorboard_dir
        self.tensorboard_path = os.path.join(self.log_root, self.time_of_run)
        self.eval_dir = os.path.join(self.tensorboard_path, "eval")
        self.test_dir = os.path.join(self.tensorboard_path, "test")
        self.__init_folder()
        self.current_exp_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        self.writer = SummaryWriter(self.tensorboard_path)
        self.best_eval_metrics = {}
        self.__init_metrics()
        self.metric_df = []
        self.eval_metric_df = []
        self.best_model_path = ""
        self.previous_best_model_path = ""
        self.best_eval_epoch_idx = None
        self.best_eval_epoch_idx_list = []
        self.test_metric_df = []
        self.master_kpi_path = master_kpi_path  # master kpi path is the epoch level summary
        write_arguments_to_file(args, os.path.join(self.tensorboard_path, "args.csv"))
        self.mlflow_log_param(args)

    def __init_folder(self):
        if not os.path.exists(self.log_root):
            Path(self.log_root).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(self.eval_dir):
            Path(self.eval_dir).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(self.test_dir):
            Path(self.test_dir).mkdir(parents=True, exist_ok=True)
        print(" Launch TensorBoard with: tensorboard --logdir=%s" % self.tensorboard_path)

    def __init_metrics(self):
        self.best_eval_metrics = {'accuracy': 0,  'macro_f1': 0}

    def reset_best_eval_metrics_models(self):
        self.__init_metrics()
        self.best_model_path = ""
        self.best_eval_epoch_idx = None
        self.previous_best_model_path = ""

    @staticmethod
    def mlflow_log_param(args):
        for key, value in vars(args).items():
            mlflow.log_param(key, value)

    def __log_scalar(self, name, value, step):
        """Log a scalar value to both MLflow and TensorBoard"""
        self.writer.add_scalar(name, value, step)
        mlflow.log_metric(name, np.double(value), step=step)

    def log_train_fold_epoch(self, y_gt, y_pred, losses_2_record, fold_idx, dataset_len, epoch_idx, batch_idx):
        """
        record the training losses and metrics, it doesn't save it to csv file only tensorboard and mlflow
        Parameters
        ----------
        y_gt: ground truth
        y_pred: predictions 1D vector
        losses_2_record: all losses of interested
        fold_idx : the fold num of xross validation
        dataset_len: the length of dataloader
        epoch_idx: epoch index e.g. 100 epochs
        batch_idx: the batch index when the interval to record is satisfied
        ------
        """
        # log all metrics of training fro tensorboard
        metrics = calc_metrics(y_gt, y_pred, avg_method=self.avg_method)
        step_idx = epoch_idx * dataset_len + batch_idx
        for metric_name, metric_value in metrics.items():
            self.__log_scalar(name="fold_%s/epoch_%s/train/%s" % (fold_idx, epoch_idx, metric_name),
                              value=metric_value, step=step_idx)
        for key, val in losses_2_record.items():
            self.__log_scalar(name="fold_%s/epoch_%s/train/%s" % (fold_idx, epoch_idx, key), value=val, step=step_idx)
        print("fold: %s, epoch: %s/%s,step [%s/%s] loss: %s, training metrics: %s"
              % (fold_idx, epoch_idx + 1, self.args.epochs, batch_idx, dataset_len, str(losses_2_record),
                 ["%s: %.3f" % (key, val) for key, val in sorted(metrics.items(), key=lambda x: x[0])]))

    def log_eval_fold_epoch(self, y_gt, y_pred, losses_2_record, fold_idx, epoch_idx, model, key_metric='macro_f1'):
        """
        record the training losses and metrics, it doesn't save it to csv file only tensorboard and mlflow
        Parameters
        ----------
        y_gt: ground truth
        y_pred: predictions 1D vector
        losses_2_record: all losses of interested
        fold_idx : the fold num of xross validation
        epoch_idx: epoch index e.g. 100 epochs
        model : pytorch model
        key_metric: metric that used to save the model
        ------
        """
        # log all metrics for eval
        metrics = calc_metrics(y_gt, y_pred, avg_method=self.avg_method)

        print("fold_%s, epoch: %s/%s, eval loss: %s, eval metrics: %s" %
              (fold_idx, epoch_idx + 1, self.args.epochs, losses_2_record,
               {"%s: %.3f" % (key, val) for key, val in sorted(metrics.items(), key=lambda x: x[0])}))
        for key, val in {**losses_2_record, **metrics}.items():  # merge them then log in one line code
            self.__log_scalar(name="fold_%s/epoch_%s/eval/%s" % (fold_idx, epoch_idx, key), value=val, step=epoch_idx)

        if metrics[key_metric] > self.best_eval_metrics[key_metric]:
            self.best_eval_metrics = metrics
            for key, val in {**losses_2_record, **metrics}.items():  # merge them then log in one line code
                self.__log_scalar(name="fold_%s/epoch_%s/eval/best_%s" % (fold_idx, epoch_idx, key),
                                  value=val, step=epoch_idx)
            # here we only save the last epoch test results from each fold
            model_save_dir = os.path.join(self.tensorboard_path, "saved_models")
            if not os.path.exists(model_save_dir):
                Path(model_save_dir).mkdir(parents=True, exist_ok=True)
            self.best_model_path = os.path.join(model_save_dir, "fold_%s_epoch_%s.pth" % (fold_idx, epoch_idx))
            torch.save(model, self.best_model_path)
            if os.path.exists(self.previous_best_model_path):
                os.remove(self.previous_best_model_path)
            self.previous_best_model_path = self.best_model_path
            self.best_eval_epoch_idx = epoch_idx
            self.best_eval_epoch_idx_list.append(epoch_idx)
        print("current best eval model index: %s" % self.best_eval_epoch_idx)
        print("fold: %s, epoch: %s/%s, best eval loss: %s, current best eval metrics: %s"
              % (fold_idx, epoch_idx + 1, self.args.epochs, str(losses_2_record),
                 ["%s: %.3f" % (key, val) for key, val in sorted(self.best_eval_metrics.items(), key=lambda x: x[0])]))
        to_json = {'fold_num': [fold_idx],
                   'epoch_num': [epoch_idx],
                   'type': ['eval'],
                   'macro_accuracy': [metrics['macro_accuracy']],
                   'macro_precision': [metrics['macro_precision']],
                   'macro_recall': [metrics['macro_recall']],
                   'macro_specificity': [metrics['macro_specificity']],
                   'macro_cohen': [metrics['macro_cohen']],
                   'best_macro_accuracy': [self.best_eval_metrics['macro_accuracy']],
                   'best_macro_f1': [self.best_eval_metrics['macro_f1']]}
        self.__write_metrics(df_2_write=pd.DataFrame.from_dict(to_json),
                             path=os.path.join(self.tensorboard_path, "eval_metrics.csv"))

    def load_best_eval_model(self, current_model, model_dir=None):
        if model_dir is None:
            model = load_torch_model_param(current_model=current_model, model_path=self.best_model_path)
        else:
            model = load_torch_model_param(current_model=current_model, model_path=model_dir)
        return model

    @staticmethod
    def __write_metrics(df_2_write, path):
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = pd.concat([df, df_2_write], axis=0, ignore_index=True)
        else:
            df = df_2_write
        df.to_csv(path, index=False)

    def log_test_fold_epoch(self, fold_idx, epoch_idx, y_gt, y_pred, losses_2_record):
        # log all metrics for testing
        test_metrics = calc_metrics(y_gt, y_pred, avg_method=self.avg_method)
        for key, val in {**losses_2_record, **test_metrics}.items():  # merge them then log in one line code
            self.__log_scalar(name="fold_%s/test/%s" % (fold_idx, key), value=val, step=epoch_idx)

        print("fold_idx: %s, best model epoch: %s, test loss: %s, testing metrics: %s"
              % (fold_idx, self.best_eval_epoch_idx, str(losses_2_record),
                 ["%s: %.3f" % (key, val) for key, val in sorted(test_metrics.items(), key=lambda x: x[0])]))
        to_json = {'fold_num': [fold_idx],
                   'epoch_num': [epoch_idx],
                   'type': ['test'],
                   'macro_accuracy': [test_metrics['macro_accuracy']],
                   'macro_precision': [test_metrics["macro_precision"]],
                   'macro_recall': [test_metrics["macro_recall"]],
                   'macro_cohen': [test_metrics['macro_cohen']],
                   'macro_f1': [test_metrics['macro_f1']]
                   }
        # cache test results
        self.__write_metrics(df_2_write=pd.DataFrame.from_dict(to_json),
                             path=os.path.join(self.tensorboard_path, "test_metrics.csv"))

    def reg_test_score_to_leaderboard(self, y_gt, y_pred, df_test, summary_folder_dic):
        # these are the experiment summary csv file
        metrics = calc_metrics(y_gt, y_pred, avg_method=self.avg_method)
        results_series = pd.Series(
            {**self.args.__dict__, **metrics,  # **{"fold_idx": fold_idx, "epoch_idx": epoch_idx},
             **{'tf': str(self.time_of_run), "machine": platform.uname()[1]},
                "best epochs": np.mean(self.best_eval_epoch_idx_list)
             })
        summary_results = pd.DataFrame(results_series).transpose()
        if os.path.exists(self.master_kpi_path):
            previous_results_df = pd.read_csv(self.master_kpi_path)
            summary_results = pd.concat([previous_results_df, summary_results], axis=0, ignore_index=True)
        summary_results.drop_duplicates(inplace=True)
        summary_results.sort_index(axis=1, ascending=False, inplace=True)
        summary_results.sort_values(by=['dataset', 'tf'], ascending=[False, True], inplace=True)
        summary_results.to_csv(self.master_kpi_path, index=False)

        # let's do the evaluation on the sp and rp
        if self.args.dataset == "apple":
            df_test = df_test.rename(columns={"appleid": "pid", "linetime": "line"})
        else:
            df_test = df_test.rename(columns={"mesaid": "pid"})

        for eval_period, summary_path in summary_folder_dic.items():
            clf_metric_summary, min_sum, label_level_sum, epoch_sleep_metrics = \
                evaluate_sleep_alg(self.tensorboard_path, df_test, num_classes=3, algorithm_name=self.args.nn_type,
                                   recording_period=eval_period, feature_type=self.args.feature_type)
            # 0: clf_metric_sum, 1: min_sum, 2: label_level_sum,
            clf_metric_summary["best epochs"] = np.mean(self.best_eval_epoch_idx_list)
            save_evaluation_summary(clf_metric_summary, min_sum, epoch_sleep_metrics, self.args, summary_path,
                                    period=eval_period, tf=self.time_of_run)
        print("all models have been evaluated")

        return

    def save_test_analysis_visualisation_results(self, y_gt, y_hat, feature, epoch=0, run_type='eval', fold_num=0):
        """
        This is a customized function for sleep analysis
        """
        # !TODO save the confusion matrix, classification report, T-SNE plot, entropy statistics
        label_values, target_names = sleep_class_name_mapping(self.args.num_classes)
        if len(y_gt.shape) > 2:
            y_gt = np.reshape(y_gt, -1)
        matrix = confusion_matrix(y_gt, y_hat)
        report = classification_report(y_gt, y_hat, target_names=target_names, digits=4)
        print("Classification report: \n")
        print(report)
        if run_type == 'eval':
            save_file_path = self.eval_dir
        else:
            save_file_path = self.test_dir
        file_title = "_%s_fold_%s_epoch_%s_" % (run_type, fold_num, epoch)
        np.savetxt(os.path.join(save_file_path, file_title + 'confusion_matrix.txt'), matrix, fmt='%d', delimiter=',')
        with open(os.path.join(save_file_path, file_title + "classification_report.txt"), "w") as text_file:
            text_file.write(report)
        # pd.DataFrame({"gt": y_gt, "pred": y_hat, "run_type": [run_type]*len(y_gt)}).to_csv(
        #     os.path.join(self.tensorboard_path, "%s_%s_prediction.csv" % (self.args.dataset, self.args.nn_type)))
        # save the best trained model as well.
        plot_save_confusion_matrix(y_gt, y_hat, normalize=True, class_names=target_names,
                                   location=save_file_path, title=file_title)
        if feature is not None:
            generate_tsne(feature, self.args.num_classes, gt=y_gt[:feature.shape[0]],
                          output_path=save_file_path, title="%s_num_classes_%s_fold_%s_epoch_%s" %
                                                            (run_type,  self.args.num_classes,
                                                             fold_num, epoch))
        return None

    def append_and_save_test_prediction(self, y_gt, y_hat, y_pred_prob, test_fold_idx, df_test):
        """
        df_test is the test dataset.
        """
        num_classes = y_pred_prob.shape[1]
        extra_column_to_save = [self.args.nn_type]
        df = pd.DataFrame({self.args.nn_type: y_hat, 'gt': y_gt, "window_idx": test_fold_idx})
        for class_label in np.arange(num_classes):
            df[class_label] = y_pred_prob[:, class_label]
            extra_column_to_save.append(class_label)
        #!TODO we need refactor this code to not hard code dataset name

        if self.args.dataset == "apple":
            df = pd.merge(left=df_test, right=df, on="window_idx")
        else:
            df_test.reset_index(inplace=True, drop=True)  # the original df_test has the index which cause misalignment
            df = pd.concat([df_test, df], axis=1)
        df['chk'] = (df['stages'] - df['gt']).abs()
        assert df['chk'].sum() == 0, print("ground truth misaligned!")
        df = df[['pid', 'stages', 'line', 'gt_sleep_block'] + extra_column_to_save]
        df.to_csv(os.path.join(self.tensorboard_path, '%s_stages_30s_%s_100_%s.csv' %
                               (self.args.num_classes, self.args.nn_type, self.args.feature_type)), index=False)
        return df

    def copy_py_files(self, files_path):
        files = get_all_files_include_sub(files_path, ".py")
        with ZipFile(os.path.join(self.tensorboard_path, time.strftime("%Y-%m-%d_%H%M") + '_' + "model_bak.zip"),
                     'w') as zipObj:
            for file in files:
                zipObj.write(file, arcname=os.path.basename(file))
                # file_name = time.strftime("%Y-%m-%d_%H%M") + '_' + os.path.basename(file)
                # shutil.copy(file, os.path.join(self.tensorboard_path, file_name))

    def copy_main_run_file(self, file):
        copied_script_name = time.strftime("%Y-%m-%d_%H%M") + '_' + os.path.basename(file) + ".zip"
        to_file_name = os.path.join(self.tensorboard_path, copied_script_name)
        with ZipFile(to_file_name, "w") as zipObj:
            zipObj.write(file, arcname=os.path.basename(file))
        # shutil.copy(file, os.path.join(self.tensorboard_path, copied_script_name))


class ClassificationTracker(object):
    """
    final results:
    args columns, best eval epoch idx, best eval acc, best eval micro_f1, best eval macro_f1, best test acc,
    for each experiment we record:
        1. the args used for the experiment
        2. append record eval and test performance after each training epoch
        3. append the summary performance to a file.
        4. save the best evaluation model.
        5. copy all model files to tensorboard folder.
    """

    def __init__(self, args, tensorboard_dir, master_kpi_path, avg_method="macro"):
        self.args = args
        self.avg_method = avg_method
        # ******** setup the tensor board path
        self.time_of_run = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_root = tensorboard_dir
        self.tensorboard_path = os.path.join(self.log_root, self.time_of_run)
        self.eval_dir = os.path.join(self.tensorboard_path, "eval")
        self.test_dir = os.path.join(self.tensorboard_path, "test")
        self.__init_folder()
        self.current_exp_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        self.writer = None
        self.best_eval_metrics = {}
        self.__init_metrics()
        self.metric_df = []
        self.eval_metric_df = []
        self.best_model_path = ""
        self.previous_best_model_path = ""
        self.best_eval_epoch_idx = None
        self.best_eval_epoch_idx_list = []
        self.test_metric_df = []
        self.master_kpi_path = master_kpi_path  # master kpi path is the epoch level summary
        write_arguments_to_file(args, os.path.join(self.tensorboard_path, "args.csv"))

    def __init_folder(self):
        if not os.path.exists(self.eval_dir):
            Path(self.eval_dir).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(self.test_dir):
            Path(self.test_dir).mkdir(parents=True, exist_ok=True)
        Path(self.tensorboard_path).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(self.log_root):
            Path(self.log_root).mkdir(parents=True, exist_ok=True)
        print(" Launch TensorBoard with: tensorboard --logdir=%s" % self.tensorboard_path)

    def __init_metrics(self):
        self.best_eval_metrics = {'accuracy': 0,  'macro_f1': 0}

    def reset_best_eval_metrics_models(self):
        self.__init_metrics()
        self.best_model_path = ""
        self.best_eval_epoch_idx = 0
        self.previous_best_model_path = ""

    def __log_scalar(self, name, value, step):
        """Log a scalar value to both MLflow and TensorBoard"""
        self.writer.add_scalar(name, value, step)

    def __setup_fold_train_val_test(self, fold_index):
        fold_dir = os.path.join(self.tensorboard_path, f"fold_{fold_index}")
        Path(fold_dir).mkdir(parents=True, exist_ok=True)
        if self.writer and self.current_fold != fold_index:  # writer is not for the current fold
            self.writer.close()
        elif not self.writer:  # or the writer is not existed
            self.current_fold = fold_index
            self.writer = SummaryWriter(fold_dir)

    def log_train_fold_epoch(self, y_gt, y_pred, losses_2_record, fold_idx, dataset_len, epoch_idx, batch_idx):
        """
        record the training losses and metrics, it doesn't save it to csv file only tensorboard and mlflow
        Parameters
        ----------
        y_gt: ground truth
        y_pred: predictions 1D vector
        losses_2_record: all losses of interested
        fold_idx : the fold num of xross validation
        dataset_len: the length of dataloader
        epoch_idx: epoch index e.g. 100 epochs
        batch_idx: the batch index when the interval to record is satisfied
        ------
        """
        # log all metrics of training fro tensorboard
        self.__setup_fold_train_val_test(fold_idx)
        metrics = calc_metrics(y_gt, y_pred, avg_method=self.avg_method)
        step_idx = epoch_idx * 10 ** len(str(dataset_len)) + batch_idx
        for metric_name, metric_value in metrics.items():
            self.__log_scalar(name=f"{metric_name}/train", value=metric_value, step=step_idx)
        for key, val in losses_2_record.items():
            self.__log_scalar(name=f"{key}/train", value=val, step=step_idx)
        print("fold: %s, epoch: %s/%s,step [%s/%s] loss: %s, training metrics: %s"
              % (fold_idx, epoch_idx + 1, self.args.epochs, batch_idx, dataset_len, str(losses_2_record),
                 ["%s: %.3f" % (key, val) for key, val in sorted(metrics.items(), key=lambda x: x[0])]))

    def log_eval_fold_epoch(self, y_gt, y_pred, losses_2_record, fold_idx, epoch_idx, model, key_metric='macro_f1'):
        """
        record the training losses and metrics, it doesn't save it to csv file only tensorboard and mlflow
        Parameters
        ----------
        y_gt: ground truth
        y_pred: predictions 1D vector
        losses_2_record: all losses of interested
        fold_idx : the fold num of xross validation
        epoch_idx: epoch index e.g. 100 epochs
        model : pytorch model
        key_metric: metric that used to save the model
        ------
        """
        # log all metrics for eval
        self.__setup_fold_train_val_test(fold_idx)
        metrics = calc_metrics(y_gt, y_pred, avg_method=self.avg_method)

        print("fold_%s, epoch: %s/%s, eval loss: %s, eval metrics: %s" %
              (fold_idx, epoch_idx + 1, self.args.epochs, losses_2_record,
               {"%s: %.3f" % (key, val) for key, val in sorted(metrics.items(), key=lambda x: x[0])}))
        for key, value in {**losses_2_record, **metrics}.items():  # merge them then log in one line code
            self.__log_scalar(name=f"{key}/val", value=value, step=epoch_idx)
        if metrics[key_metric] > self.best_eval_metrics[key_metric]:
            self.best_eval_metrics = metrics
            for key, value in {**losses_2_record, **metrics}.items():  # merge them then log in one line code
                self.__log_scalar(name=f"{key}/val", value=value, step=epoch_idx)
            # here we only save the last epoch test results from each fold
            model_save_dir = os.path.join(self.tensorboard_path, "saved_models")
            if not os.path.exists(model_save_dir):
                Path(model_save_dir).mkdir(parents=True, exist_ok=True)
            self.best_model_path = os.path.join(model_save_dir, "fold_%s_epoch_%s.pth" % (fold_idx, epoch_idx))
            torch.save(model, self.best_model_path)
            if os.path.exists(self.previous_best_model_path):
                os.remove(self.previous_best_model_path)
            self.previous_best_model_path = self.best_model_path
            self.best_eval_epoch_idx = epoch_idx
            self.best_eval_epoch_idx_list.append(epoch_idx)
        print("current best eval model index: %s" % self.best_eval_epoch_idx)
        print("fold: %s, epoch: %s/%s, best eval loss: %s, current best eval metrics: %s"
              % (fold_idx, epoch_idx + 1, self.args.epochs, str(losses_2_record),
                 ["%s: %.3f" % (key, val) for key, val in sorted(self.best_eval_metrics.items(), key=lambda x: x[0])]))
        to_json = {'fold_num': [fold_idx],
                   'epoch_num': [epoch_idx],
                   'type': ['eval'],
                   'macro_accuracy': [metrics['macro_accuracy']],
                   'macro_precision': [metrics['macro_precision']],
                   'macro_recall': [metrics['macro_recall']],
                   'macro_specificity': [metrics['macro_specificity']],
                   'macro_cohen': [metrics['macro_cohen']],
                   'best_macro_accuracy': [self.best_eval_metrics['macro_accuracy']],
                   'best_macro_f1': [self.best_eval_metrics['macro_f1']]}
        self.__write_metrics(df_2_write=pd.DataFrame.from_dict(to_json),
                             path=os.path.join(self.tensorboard_path, "eval_metrics.csv"))

    def load_best_eval_model(self, current_model, model_dir=None):
        if model_dir is None:
            model = load_torch_model_param(current_model=current_model, model_path=self.best_model_path)
        else:
            model = load_torch_model_param(current_model=current_model, model_path=model_dir)
        return model

    @staticmethod
    def __write_metrics(df_2_write, path):
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = pd.concat([df, df_2_write], axis=0, ignore_index=True)
        else:
            df = df_2_write
        df.to_csv(path, index=False)

    def log_test_fold_epoch(self, fold_idx, epoch_idx, y_gt, y_pred, losses_2_record):
        # log all metrics for testing
        test_metrics = calc_metrics(y_gt, y_pred, avg_method=self.avg_method)
        for key, value in {**losses_2_record, **test_metrics}.items():  # merge them then log in one line code
            self.__log_scalar(name=f"{key}/test", value=value, step=epoch_idx)

        print("fold_idx: %s, best model epoch: %s, test loss: %s, testing metrics: %s"
              % (fold_idx, self.best_eval_epoch_idx, str(losses_2_record),
                 ["%s: %.3f" % (key, val) for key, val in sorted(test_metrics.items(), key=lambda x: x[0])]))
        to_json = {'fold_num': [fold_idx],
                   'epoch_num': [epoch_idx],
                   'type': ['test'],
                   'macro_accuracy': [test_metrics['macro_accuracy']],
                   'macro_precision': [test_metrics["macro_precision"]],
                   'macro_recall': [test_metrics["macro_recall"]],
                   'macro_cohen': [test_metrics['macro_cohen']],
                   'macro_f1': [test_metrics['macro_f1']]
                   }
        # cache test results
        self.__write_metrics(df_2_write=pd.DataFrame.from_dict(to_json),
                             path=os.path.join(self.tensorboard_path, "test_metrics.csv"))

    def reg_test_score_to_leaderboard(self, y_gt, y_pred, df_test, summary_folder_dic):
        # these are the experiment summary csv file
        metrics = calc_metrics(y_gt, y_pred, avg_method=self.avg_method)
        results_series = pd.Series(
            {**self.args.__dict__, **metrics,  # **{"fold_idx": fold_idx, "epoch_idx": epoch_idx},
             **{'tf': str(self.time_of_run), "machine": platform.uname()[1]},
             "best epochs": self.best_eval_epoch_idx_list[-1]
             })
        summary_results = pd.DataFrame(results_series).transpose()
        if os.path.exists(self.master_kpi_path):
            previous_results_df = pd.read_csv(self.master_kpi_path)
            summary_results = pd.concat([previous_results_df, summary_results], axis=0, ignore_index=True)
        summary_results.drop_duplicates(inplace=True)
        summary_results.sort_index(axis=1, ascending=False, inplace=True)
        summary_results.sort_values(by=['dataset', 'tf'], ascending=[False, True], inplace=True)
        summary_results.to_csv(self.master_kpi_path, index=False)

        # let's do the evaluation on the sp and rp
        if self.args.dataset == "apple":
            df_test = df_test.rename(columns={"appleid": "pid", "linetime": "line"})
        else:
            df_test = df_test.rename(columns={"mesaid": "pid"})

        for eval_period, summary_path in summary_folder_dic.items():
            clf_metric_summary, min_sum, label_level_sum, epoch_sleep_metrics = \
                evaluate_sleep_alg(self.tensorboard_path, df_test, num_classes=3, algorithm_name=self.args.nn_type,
                                   recording_period=eval_period, feature_type=self.args.feature_type)
            # 0: clf_metric_sum, 1: min_sum, 2: label_level_sum,
            clf_metric_summary["best epochs"] = self.best_eval_epoch_idx_list[-1]
            save_evaluation_summary(clf_metric_summary, min_sum, epoch_sleep_metrics, self.args, summary_path,
                                    period=eval_period, tf=self.time_of_run)
        print("all models have been evaluated")

        return

    def save_test_analysis_visualisation_results(self, y_gt, y_hat, feature, epoch=0, run_type='eval', fold_num=0):
        """
        This is a customized function for sleep analysis
        """
        # !TODO save the confusion matrix, classification report, T-SNE plot, entropy statistics
        label_values, target_names = sleep_class_name_mapping(self.args.num_classes)
        if len(y_gt.shape) > 2:
            y_gt = np.reshape(y_gt, -1)
        matrix = confusion_matrix(y_gt, y_hat)
        report = classification_report(y_gt, y_hat, target_names=target_names, digits=4)
        print("Classification report: \n")
        print(report)
        if run_type == 'eval':
            save_file_path = self.eval_dir
        else:
            save_file_path = self.test_dir
        file_title = "_%s_fold_%s_epoch_%s_" % (run_type, fold_num, epoch)
        np.savetxt(os.path.join(save_file_path, file_title + 'confusion_matrix.txt'), matrix, fmt='%d', delimiter=',')
        with open(os.path.join(save_file_path, file_title + "classification_report.txt"), "w") as text_file:
            text_file.write(report)
        # pd.DataFrame({"gt": y_gt, "pred": y_hat, "run_type": [run_type]*len(y_gt)}).to_csv(
        #     os.path.join(self.tensorboard_path, "%s_%s_prediction.csv" % (self.args.dataset, self.args.nn_type)))
        # save the best trained model as well.
        plot_save_confusion_matrix(y_gt, y_hat, normalize=True, class_names=target_names,
                                   location=save_file_path, title=file_title)
        if feature is not None:
            generate_tsne(feature, self.args.num_classes, gt=y_gt[:feature.shape[0]],
                          output_path=save_file_path, title="%s_num_classes_%s_fold_%s_epoch_%s" %
                                                            (run_type,  self.args.num_classes,
                                                             fold_num, epoch))
        return None

    def append_and_save_test_prediction(self, y_gt, y_hat, y_pred_prob, test_fold_idx, df_test):
        """
        df_test is the test dataset.
        """
        num_classes = y_pred_prob.shape[1]
        extra_column_to_save = [self.args.nn_type]
        df = pd.DataFrame({self.args.nn_type: y_hat, 'gt': y_gt, "window_idx": test_fold_idx})
        for class_label in np.arange(num_classes):
            df[class_label] = y_pred_prob[:, class_label]
            extra_column_to_save.append(class_label)
        #!TODO we need refactor this code to not hard code dataset name

        if self.args.dataset == "apple":
            df = pd.merge(left=df_test, right=df, on="window_idx")
        else:
            df_test.reset_index(inplace=True, drop=True)  # the original df_test has the index which cause misalignment
            df = pd.concat([df_test, df], axis=1)
        df['chk'] = (df['stages'] - df['gt']).abs()
        assert df['chk'].sum() == 0, print("ground truth misaligned!")
        df = df[['pid', 'stages', 'line', 'gt_sleep_block'] + extra_column_to_save]
        df.to_csv(os.path.join(self.tensorboard_path, '%s_stages_30s_%s_100_%s.csv' %
                               (self.args.num_classes, self.args.nn_type, self.args.feature_type)), index=False)
        return df

    def copy_py_files(self, files_path):
        files = get_all_files_include_sub(files_path, ".py")
        with ZipFile(os.path.join(self.tensorboard_path, time.strftime("%Y-%m-%d_%H%M") + '_' + "model_bak.zip"),
                     'w') as zipObj:
            for file in files:
                zipObj.write(file, arcname=os.path.basename(file))
                # file_name = time.strftime("%Y-%m-%d_%H%M") + '_' + os.path.basename(file)
                # shutil.copy(file, os.path.join(self.tensorboard_path, file_name))

    def copy_main_run_file(self, file):
        copied_script_name = time.strftime("%Y-%m-%d_%H%M") + '_' + os.path.basename(file) + ".zip"
        to_file_name = os.path.join(self.tensorboard_path, copied_script_name)
        with ZipFile(to_file_name, "w") as zipObj:
            zipObj.write(file, arcname=os.path.basename(file))
        # shutil.copy(file, os.path.join(self.tensorboard_path, copied_script_name))


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
