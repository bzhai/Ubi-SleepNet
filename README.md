# Ubi-SleepNet
This is the code repository for paper: Ubi-SleepNet: Advanced Multimodal Fusion Techniques for
Three-stage Sleep Classification Using Ubiquitous Sensing.

## Dataset Download
* [MESA Dataset](https://sleepdata.org/datasets/mesa)
* [Apple Watch](https://physionet.org/content/sleep-accel/1.0.0/heart_rate/)

## Dataset Building
* Data Pre-processing code is available at:[MakeSenseOfSleep](https://github.com/bzhai/multimodal_sleep_stage_benchmark) which should produce a h5 file for MESA dataset.
* This repository includes the builders for MESA statistic features and Apple Watch dataset with statistic features
* For the MESA with HRV feature set, please go to [MakeSenseOfSleep](https://github.com/bzhai/multimodal_sleep_stage_benchmark). It includes a dataset builder that can build the HRV feature dataset.
* This repository also includes a data builder to build the dataset that uses the raw accelerometer 
and HR data collected from the Apple Watch dataset.

## Set up environment
To ensure the experiments run smoothly, please create a **python 3.8** environment, and please be aware, the `pytables` and `h5py` requires to be installed via `conda` . 

## Running Experiments
you could run a non-attention based model by:

    python -m train_val_test --nn_type Vggacc79f174_7 --epochs 20 --dataset mesa

To run the attention model, the modality should be specified. The code below is an example:

    python -m train_val_test --nn_type VggAcc79F174_SplitModal_SANTimeDimMatrixAttOnMod1NLayer1 --epochs 20 --dataset mesa --att_on_modality act

