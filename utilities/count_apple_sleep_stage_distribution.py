import h5py as h5py
import h5py as h5py
from sleep_stage_config import Config
from collections import Counter

cfg =Config()
cache_path = cfg.APPLE_LOOCV_ALL_WINDOWED % 100
with h5py.File(cache_path, 'r') as data:
    df_data = data["df_values"][:]
    x = data["x"][:]
    y = data["y"][:]
    columns = data["columns"][:].astype(str).tolist()
    data.close()
y = list(y)

print(Counter(y))