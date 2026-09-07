# %%
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from IPython.core.pylabtools import figsize

sys.path.append('..//')
from utils_mitgcm import open_mitgcm_ds_from_config
from utils_energy_analysis import *

# %%
lake = 'zug'
model = f'{lake}_2025'
mitgcm_config, ds = open_mitgcm_ds_from_config('..//config.json', model)

# %%
folder_path = os.path.dirname(mitgcm_config['datapath'])
output_folder = os.path.join(folder_path, "chaos_analysis")
os.makedirs(output_folder, exist_ok=True)

# %%
grid_resolution = 100
ds['YC'] = np.arange(1, len(ds['YC']) + 1) * grid_resolution - grid_resolution / 2
ds['XC'] = np.arange(1, len(ds['XC']) + 1) * grid_resolution - grid_resolution / 2
ds['YG'] = np.arange(0, len(ds['YG'])) * grid_resolution
ds['XG'] = np.arange(0, len(ds['XG'])) * grid_resolution

# %%
mask = ds['THETA'].isel(time=0).values > 0

# %%
z_lim=16
list_idx_chaos = []
for tt in range(len(ds['time'])):
    idx_chaos=0
    for zz in range(z_lim):
        u_sign_change = np.sum(np.diff(np.sign(ds['UVEL'].isel(time=tt, Z=zz).values)) != 0)
        v_sign_change = np.sum(np.diff(np.sign(ds['VVEL'].isel(time=tt, Z=zz).values)) != 0)
        idx_chaos += (u_sign_change + v_sign_change) / np.sum(mask[:z_lim])
    list_idx_chaos.append(idx_chaos)

import pandas as pd

df = pd.DataFrame({
    "time": ds["time"].values,
    "chaos_index": list_idx_chaos
})

csv_path = os.path.join(output_folder, "chaos_index.csv")
df.to_csv(csv_path, index=False)

print(f"Saved to: {csv_path}")


