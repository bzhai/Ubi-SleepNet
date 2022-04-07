import argparse

from data_loader.TorchFrameDataLoader import get_apple_loocv_ids
from utilities.utils import *
from sleep_stage_config import Config
import time
from copy import copy
from models.build_raw_model import build_raw_model

from data_loader.raw_data_loader import *
import torch
import torch.nn as nn
from torchsummary import summary
import sys
from pathlib import Path
import mlflow
import mlflow.pytorch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import f1_score
import shutil
from utilities.tracker_utils import ClassificationTracker
from scipy.special import softmax

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def get_total_folds_raw_data(dataset_name):
    """
    return total folds based on the dataset name
    """
    if dataset_name == "mesa_raw":
        total_fold = 1
    elif dataset_name == "apple_raw":
        total_fold = 16
    else:
        raise ValueError("Dataset is not recognised")
    return total_fold


def main(args):
    setup_seed(args.seed)
    
    cfg = Config()
    total_folds = get_total_folds_raw_data(args.dataset)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    log_root = os.path.join(r"tfboard", args.dataset, args.nn_type)
    tracker = ClassificationTracker(args, tensorboard_dir=log_root, master_kpi_path="./exp_results.csv")
    sys.stdout = Logger(tracker.tensorboard_path)

    print("Model: %s" % args.nn_type)
    print(" Launch TensorBoard with: tensorboard --logdir=%s" % tracker.tensorboard_path)
    print_args(args)

    tracker.copy_main_run_file(os.path.join(os.path.abspath(os.getcwd()), os.path.basename(__file__)))
    tracker.copy_py_files(os.path.join(os.path.abspath(os.getcwd()), "models"))
    test_fold_gt = []  # this list can be used for hold out and CV
    test_fold_pred = []
    test_fold_prob = []
    test_fold_feature = []
    test_fold_idx = []
    df_test = []
    # df_test = get_test_df(cfg=cfg, dataset=args.dataset, num_classes=args.num_classes, seq_len=args.seq_len)
    for fold_num in np.arange(total_folds):

        model = build_raw_model(args.nn_type, args.dataset, args.seq_len, args.num_classes,
                                modality=args.att_on_modality)
        if torch.cuda.is_available():
            model.cuda()
        optims = {"SGD": torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum),
                  # "ADAM":torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                  #                         eps=1e-08, weight_decay=0, amsgrad=False),
                  "ADAM": torch.optim.Adam(model.parameters(), lr=args.lr)
                  }
        optimizer = optims[args.optim]
        # summary(model, [(3, (int(args.seq_len) + 1) * 30), (6, int(args.seq_len) + 1)] )  # print the model summary
        # we need to get the train val test pid
        train_id_list, val_id_list, test_id_list = get_apple_loocv_ids(cfg, fold_num)
        print("start training process !")
        # delete
        # model = tracker.load_best_eval_model(model, 'after2021-02-24_train_val_test/tfboard\\mesa_raw\\VggAcc79F174_7_RM_RAW\\20210420-230416\\saved_models\\fold_0_epoch_0.pth')
        # tracker.reset_best_eval_metrics_models()
        for epoch in range(args.epochs):
            # ***************** Training ************************
            # control the accumulated samples that added to gt and pred to calculate metrics at the last batch
            # train_loader, test_loader, val_loader = \
            #     get_windowed_train_test_val_loader(cfg=cfg, batch_size=args.batch_size, seq_len=args.seq_len,
            #                                        num_classes=args.num_classes, dataset=args.dataset, fold=fold_num)
            first_train_epoch = True
            train_num = 1
            if args.debug == 1:
                train_id_list = train_id_list[:2]
            for train_pid in train_id_list:
                train_loader = get_raw_dataloader_by_id(train_pid, cfg=cfg, shuffle=True,
                                                        batch_size=args.batch_size, data_set=args.dataset,
                                                        seq_len=args.seq_len, apple_acc_hz=1)
                model.train()
                for batch_idx, (acc, hrv, y, train_idx) in enumerate(train_loader):
                    acc = acc.to(device)
                    hrv = hrv.to(device)
                    y = y.to(device)
                    # Forward pass
                    outputs = model(acc, hrv)
                    if type(outputs) in (tuple, list):
                        feature, outputs = outputs[0], outputs[1]
                    else:
                        feature = outputs
                    train_loss = criterion(outputs, y)
                    # Backward and optimize
                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()
                    # Calculate the performance
                    _, predicted = torch.max(outputs.data, dim=1)
                    if first_train_epoch:
                        epoch_gt = copy(y)
                        epoch_pred = copy(predicted)
                    else:
                        epoch_gt = torch.cat([epoch_gt, y])
                        epoch_pred = torch.cat([epoch_pred, predicted])
                    first_train_epoch = False
                tracker.log_train_fold_epoch(epoch_gt.cpu().numpy(), epoch_pred.cpu().numpy(),
                                             {'xent': train_loss.item()}, fold_num, len(train_id_list), epoch,
                                             train_num)
                train_num += 1

            # ************** validation ******************
            print("validation start...")
            first_val_epoch = True
            num_val_samples = 0
            total_val_loss = 0
            val_num = 0
            # val_fc3 = []
            model.eval()
            if args.debug == 1:
                val_id_list = val_id_list[:2]
            for val_pid in val_id_list:
                val_loader = get_raw_dataloader_by_id(val_pid, cfg=cfg, shuffle=False, batch_size=args.batch_size,
                                                      data_set=args.dataset, seq_len=args.seq_len, apple_acc_hz=1)
                with torch.no_grad():
                    for batch_idx, (acc, hrv, y, val_idx) in enumerate(val_loader):
                        acc = acc.to(device)
                        hrv = hrv.to(device)
                        y = y.to(device)
                        y_outputs = model(acc, hrv)
                        if type(y_outputs) in (tuple, list):
                            _, y_val_pred = y_outputs[0], y_outputs[1]
                        else:
                            y_val_pred = y_outputs
                            # val_feature = y_outputs
                        # if batch_idx < 5:
                        #     val_fc3.append(val_feature)
                        val_loss = criterion(y_val_pred, y)
                        total_val_loss += val_loss
                        _, y_val_pred = torch.max(y_val_pred.data, dim=1)
                        num_val_samples += y.nelement()
                        if first_val_epoch:
                            val_epoch_gt = copy(y)
                            val_epoch_pred = copy(y_val_pred)
                        else:
                            val_epoch_gt = torch.cat([val_epoch_gt, y])
                            val_epoch_pred = torch.cat([val_epoch_pred, y_val_pred])
                        first_val_epoch = False
                    mean_val_loss = total_val_loss / num_val_samples
                    print(f"Number of val samples is: {num_val_samples}")
                val_num += 1
                    # val_fc3 = torch.cat(val_fc3, dim=0).cpu()
            # for each epoch run val to save the best model
            tracker.log_eval_fold_epoch(val_epoch_gt.cpu().numpy(), val_epoch_pred.cpu().numpy(),
                                        {'mean_xent': mean_val_loss.cpu().numpy()}, fold_num, epoch, model)
            # we stopped record the validation analysis results
            # tracker.save_test_analysis_visualisation_results(val_epoch_gt.cpu().numpy(),
            #                                                  val_epoch_pred.cpu().numpy(),
            #                                                  val_fc3.cpu().numpy(), epoch, 'eval',
            #                                                  fold_num=fold_num)
        # ************** test ******************
        # load the best
        print("testing start...")

        # load the best validation model
        model = tracker.load_best_eval_model(model)
        model.eval()

        for test_pid in test_id_list:
            test_loader = get_raw_dataloader_by_id(test_pid, cfg=cfg, shuffle=False, batch_size=args.batch_size,
                                                   data_set=args.dataset, seq_len=args.seq_len, apple_acc_hz=1)
            df_test_sub = get_raw_test_df(pid=test_pid, cfg=cfg, dataset=args.dataset, num_classes=args.num_classes,
                                          seq_len=args.seq_len)
            df_test.append(df_test_sub)
            first_test_epoch = True
            num_test_samples = 0
            correct_test = 0
            total_test_loss = 0
            test_fc3_feature = []
            test_idx_epoch_list = []
            test_num = 0
            with torch.no_grad():
                for batch_idx, (acc, hrv, y, test_idx) in enumerate(test_loader):
                    acc = acc.to(device)
                    hrv = hrv.to(device)
                    y = y.to(device)
                    y_outputs = model(acc, hrv)
                    if type(y_outputs) in (tuple, list):
                        test_feature, y_test_prob = y_outputs[0], y_outputs[1]
                    else:
                        y_test_prob = y_outputs
                        test_feature = y_outputs
                    if batch_idx < 5:
                        test_fc3_feature.append(test_feature)
                    test_loss = criterion(y_test_prob, y)
                    total_test_loss += test_loss
                    _, y_test_pred = torch.max(y_test_prob.data, dim=1)
                    num_test_samples += y.nelement()
                    correct_test += y_test_pred.eq(y.data).sum().item()
                    if first_test_epoch:
                        test_epoch_gt = copy(y)
                        test_epoch_pred = copy(y_test_pred)
                        test_epoch_prob = copy(y_test_prob)
                    else:
                        test_epoch_gt = torch.cat([test_epoch_gt, y])
                        test_epoch_pred = torch.cat([test_epoch_pred, y_test_pred])
                        test_epoch_prob = torch.cat([test_epoch_prob, y_test_prob])
                    first_test_epoch = False
                    test_idx_epoch_list.append(test_idx)
                mean_teat_loss = total_test_loss / num_test_samples
                test_fc3_feature = torch.cat(test_fc3_feature, dim=0).cpu()
                test_idx_epoch_list = torch.cat(test_idx_epoch_list, dim=0).cpu()
                tracker.log_test_fold_epoch(fold_num, tracker.best_eval_epoch_idx, test_epoch_gt.cpu().numpy(),
                                            test_epoch_pred.cpu().numpy(),
                                            {'mean_xent': mean_teat_loss.cpu().numpy()})
                test_fold_feature.append(test_fc3_feature.cpu().numpy())
                test_fold_gt.append(np.expand_dims(test_epoch_gt.cpu().numpy(), axis=1))
                test_fold_pred.append(np.expand_dims(test_epoch_pred.cpu().numpy(), axis=1))

                test_fold_prob.append(test_epoch_prob.cpu().numpy())
                test_fold_idx.append(np.expand_dims(test_idx_epoch_list.cpu().numpy(), axis=1))
            test_num += 1
    test_fold_idx = np.vstack(test_fold_idx).squeeze()
    test_fold_gt = np.vstack(test_fold_gt).squeeze()
    test_fold_pred = np.vstack(test_fold_pred).squeeze()
    test_fold_feature = np.vstack(test_fold_feature)
    test_fold_prob = softmax(np.vstack(test_fold_prob), axis=1)  # softmax over
    df_test = pd.concat(df_test, ignore_index=True)
    # current_df_test = df_test.copy(deep=True)
    current_df_test = tracker.append_and_save_test_prediction(test_fold_gt, test_fold_pred, test_fold_prob,
                                                              test_fold_idx, df_test=df_test)
    tracker.save_test_analysis_visualisation_results(test_fold_gt, test_fold_pred,
                                                     test_fold_feature, tracker.best_eval_epoch_idx, 'test')
    tracker.reg_test_score_to_leaderboard(y_gt=test_fold_gt, y_pred=test_fold_pred, df_test=current_df_test,
                                          summary_folder_dic=cfg.SUMMARY_FOLDER_DICT)
    print("Finished!")


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    # specialised parameters
    parser.add_argument('--nn_type', type=str, default="ResPlus_Raw_Appl1",
                        help='define the neural network type')
    parser.add_argument('--att_on_modality', type=str, default="none", help="act, car, none")
    # general parameters for all models
    parser.add_argument('--optim', type=str, default="ADAM", help='optimisation')
    parser.add_argument('--log_interval', type=int, default=100, help='interval to log metrics')
    parser.add_argument('--num_classes', type=int, default=3, help='number of classes or labels')
    parser.add_argument('--momentum', type=float, default=0.9, help='opt momentum')
    parser.add_argument('--epochs', type=int, default=1, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='training batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--debug', type=int, default=0, help='debug model')
    parser.add_argument('--dataset', type=str, default="apple_raw", help="apple, mesa, mesa_hr_statistic")
    parser.add_argument('--feature_type', type=str, default="all", help="all, hrv, hr, and so on")
    parser.add_argument('--seq_len', type=int, default=100, help="100, 50, 20")
    parser.add_argument('--comments', type=str, default="", help="comments to append")
    parser.add_argument('--seed', type=int, default=42, help="fix seed")
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
