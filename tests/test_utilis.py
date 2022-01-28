import pandas as pd
from numpy.testing import assert_almost_equal, assert_array_equal
from utilities.utils import build_windowed_data
from utilities.utils import interpolate_rri_nan_values, up_down_rri_sampling
import numpy as np


def test_interpolate_rri_nan_values():
    # Setup
    test_data = [np.nan, np.nan, 1.0, 2, 3, np.nan, 4, 5, np.nan, np.nan, 6, np.nan, np.nan, np.nan]
    # Exercise
    results = interpolate_rri_nan_values(test_data)
    # Verify
    expected = [1.0, 1.0, 1.0, 2.0, 3.0, 3.5, 4.0, 5.0, 5.33, 5.67, 6.0, 6.0, 6.0, 6.0]
    assert_almost_equal(expected, results, decimal=2)


def test_up_down_rri_sampling():
    test_data = [714.84375, 785.15625, 781.25, 769.53125, 792.96875, 789.0625, 777.34375, 785.15625]
    test_data_idx = np.asarray(pd.Series(test_data).cumsum())
    test_data = np.asarray(test_data)
    seconds_to_sample = 30
    results = up_down_rri_sampling(test_data_idx, np.asarray(test_data), seconds=seconds_to_sample, target_sampling_rate=1)
    assert results[len(test_data):].sum() == test_data.mean() * (seconds_to_sample-len(test_data))


def test_build_windowed_data():
    data = np.arange(1, 121)
    # let's make a time series with sampling rate in 1Hz and per 10 seconds as an epoch
    data = data.reshape(-1, 10)
    sampling_rate = 1
    win_len = 4
    # we should expect the return data in window length 5
    output = build_windowed_data(data, sampling_rate=sampling_rate, epoch_len=10, win_len=win_len)
    assert output.shape == (12, 50)
    # let's calculate the first 4 rows
    expected_result_1 = np.asarray([np.arange(1, 31).sum() + (-1)*20,
                                    np.arange(1, 41).sum() + (-1) * 10,
                                    np.arange(1, 51).sum(),
                                    np.arange(11, 61).sum()])
    assert_array_equal(output[:4, :].sum(axis=-1) == expected_result_1, [True, True, True, True])
    # let's calculate the rows 4-6
    expected_result_2 = np.asarray([np.arange(31, 81).sum(),
                                    np.arange(41, 91).sum(),
                                    np.arange(51, 101).sum()])
    assert_array_equal(output[5:8, :].sum(axis=-1) == expected_result_2, [True, True, True])
    # let's calculate the last two rows
    expected_result_2 = np.asarray([np.arange(81, 121).sum() + (-1)*10,
                                    np.arange(91, 121).sum() + (-1)*20])
    assert_array_equal(output[-2:, :].sum(axis=-1) == expected_result_2, [True, True])

