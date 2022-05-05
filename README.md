# Ubi-SleepNet
This is the code repository for paper: Ubi-SleepNet: Advanced Multimodal Fusion Techniques for
Three-stage Sleep Classification Using Ubiquitous Sensing.
## Updates
* 2022-04-20 Uploaded a preprocessed and windowed Apple Watch dataset based on raw accelerometer data and heart rate.
* 2022-05-05 Uploaded preprocessed CSV files of the Apple Watch dataset
## Dataset Download
* [MESA Dataset](https://sleepdata.org/datasets/mesa)
* [Apple Watch](https://physionet.org/content/sleep-accel/1.0.0/heart_rate/)
* [Preprocessed Apple Watch H5 file used in the paper](https://drive.google.com/drive/folders/1GDPVpUZMes8FZz1fieGQt0eYBEiakUke?usp=sharing) Please download the H5 files and put it into the Dataset folder
* [Preprocessed Apple Watch CSV file used in the paper](https://drive.google.com/drive/folders/1nZ3Bu0P6z_7jM-eFiNTvDmNtshMOzhZC?usp=sharing) Please download the H5 files and put it into the Dataset folder
## Models Trained in the paper:


## Dataset Building
* Data Pre-processing code is available at:[MakeSenseOfSleep](https://github.com/bzhai/multimodal_sleep_stage_benchmark) which should produce a h5 file for MESA dataset.
* This repository includes the builders for MESA statistic features and Apple Watch dataset with statistic features
* For the MESA with HRV feature set, please go to [MakeSenseOfSleep](https://github.com/bzhai/multimodal_sleep_stage_benchmark). It includes a dataset builder that can build the HRV feature dataset.
* This repository also includes a data builder to build the dataset that uses the raw accelerometer
and HR data collected from the Apple Watch dataset.
* The data processing code was adapted from [link](https://github.com/ojwalch/sleep_classifiers.git) and located under the `applewatch_dataprocessing` folder. I also updated the processed data in `outputs\features`.

## Set up environment
To ensure the experiments run smoothly, please create a **python 3.8** environment, and please be aware, the `pytables` and `h5py` requires to be installed via `conda` .

## Running Experiments
you could run a non-attention based model by:

    python -m train_val_test --nn_type Vggacc79f174_7 --epochs 20 --dataset mesa

To run the attention model, the modality should be specified. The code below is an example:

    python -m train_val_test --nn_type VggAcc79F174_SplitModal_SANTimeDimMatrixAttOnMod1NLayer1 --epochs 20 --dataset mesa --att_on_modality act

## Citation 
If you found this paper is helpful and like it. Please don't mind citing it and thank you. 
```@article{zhai2021ubi,
  title={Ubi-SleepNet: Advanced Multimodal Fusion Techniques for Three-stage Sleep Classification Using Ubiquitous Sensing},
  author={Zhai, Bing and Guan, Yu and Catt, Michael and Pl{\"o}tz, Thomas},
  journal={Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume={5},
  number={4},
  pages={1--33},
  year={2021},
  publisher={ACM New York, NY, USA}
}```
