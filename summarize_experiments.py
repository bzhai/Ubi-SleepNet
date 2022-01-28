from evaluation import *
import platform

def evaluate_sleep_alg(result_folder, combined_df, num_classes, algorithm_name, recording_period, feature_type='all'):
    """
    combined_df must has the correct classes
    """
    algs = [algorithm_name]
    summary_folder = {'r': 'rp_summary', 's': 'sp_summary'}
    if not os.path.exists(os.path.join(result_folder, summary_folder[recording_period])):
        os.mkdir(os.path.join(result_folder, summary_folder[recording_period]))
    PICKLE_RESULT_FILE = os.path.join(result_folder, summary_folder[recording_period], "%d_stages_results_%s.pkl"
                                      % (num_classes, feature_type))
    # the summary file used to save the evaluation summary
    SUMMARY_FILE = os.path.join(result_folder, summary_folder[recording_period], "%d_stages_summary_%s.csv"
                                % (num_classes, feature_type))

    LABEL_PICKLE_RESULT_FILE = os.path.join(result_folder, summary_folder[recording_period]
                                            , "%d_stages_label_level_results_%s.pkl"
                                            % (num_classes, feature_type))
    LABEL_SUMMARY_FILE = os.path.join(result_folder, summary_folder[recording_period],
                                      "%d_stages_label_level_summary_%s.csv"
                                      % (num_classes, feature_type))
    MINUTES_PREDICTION_RESULT_FILE = os.path.join(result_folder, summary_folder[recording_period]
                                                  , "%d_stages_minutes_results_%s.pkl"
                                                  % (num_classes, feature_type))
    MINUTES_PREDICTION_SUMMARY_FILE = os.path.join(result_folder, summary_folder[recording_period]
                                                   , "%d_stages_minutes_summary_%s.csv"
                                                   % (num_classes, feature_type))
    print('loading prediction results from %s' % result_folder)

    if recording_period == "s":
        df = combined_df[combined_df["gt_sleep_block"]== 1]
    else:
        df = combined_df
    print("Expanding algorithms...")

    print("start %d stages level evaluation" % num_classes)
    min_sum, mins_pred_results = evaluate_whole_period_time(df, algs, num_classes)
    min_sum = pd.DataFrame(min_sum)
    min_sum = min_sum.rename(columns=convert_int_to_label(num_classes))
    min_sum = min_sum.reindex(sorted(min_sum.columns), axis=1)
    min_sum.to_csv(MINUTES_PREDICTION_SUMMARY_FILE)
    print("Minutes summary saved to %s" % MINUTES_PREDICTION_SUMMARY_FILE)
    with open(MINUTES_PREDICTION_RESULT_FILE, "wb") as f:
        pickle.dump(mins_pred_results, f)
    print("Created minutes prediction result file to '%s'" % MINUTES_PREDICTION_RESULT_FILE)

    label_level_sum, label_level_results = label_level_evaluation_summary(df, algs, num_classes)
    label_level_sum = pd.DataFrame(label_level_sum)
    label_level_sum = label_level_sum.reindex(sorted(label_level_sum.columns), axis=1)
    label_level_sum.to_csv(LABEL_SUMMARY_FILE)
    print("Label level summary saved to %s" % LABEL_SUMMARY_FILE)
    with open(LABEL_PICKLE_RESULT_FILE, "wb") as f:
        pickle.dump(label_level_results, f)
    print("Created label level metrics result file '%s'" % LABEL_PICKLE_RESULT_FILE)

    clf_metric_summary, results = classifier_level_evaluation_summary(df, algs, eval_method="macro"
                                                           , num_classes=num_classes)  # doesn't do rescore
    clf_metric_summary = pd.DataFrame(clf_metric_summary)
    clf_metric_summary = clf_metric_summary.reindex(sorted(clf_metric_summary.columns), axis=1)
    clf_metric_summary.to_csv(SUMMARY_FILE)
    print("Classifier level summary saved to '%s'" % SUMMARY_FILE)
    with open(PICKLE_RESULT_FILE, "wb") as f:
        pickle.dump(results, f)
    print("Created classifier level result file '%s'" % PICKLE_RESULT_FILE)
    epoch_sleep_metrics = calc_metrics(df["stages"].values, df[algorithm_name].values)
    return clf_metric_summary, min_sum, label_level_sum, epoch_sleep_metrics


def save_evaluation_summary(summary, min_sum, epoch_metrics, args, exp_summary_file_path, period="s", tf=''):
    seed_df = pd.Series({"machine": platform.uname()[1]})
    seed_df['period'] = period
    seed_df['tf'] = tf
    epoch_metric_df = pd.Series(epoch_metrics)
    seed_df = seed_df.append(epoch_metric_df)
    seed_df = pd.DataFrame(seed_df.append(pd.Series({**args.__dict__}))).transpose()
    seed_df = pd.concat([seed_df, min_sum.reset_index(drop=True), summary.reset_index(drop=True)], axis=1)
    if os.path.exists(exp_summary_file_path):
        results_df = pd.read_csv(exp_summary_file_path)
        results_df = pd.concat([results_df, seed_df], axis=0, ignore_index=True)
    else:
        results_df = seed_df

    results_df.to_csv(exp_summary_file_path, index=False)


