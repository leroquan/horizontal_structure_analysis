# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
import sys

from partd import python
from pathlib import Path
import gc

sys.path.append(os.path.abspath('../../'))
from utils_mitgcm import *
from utils_modal_analysis import load_mode_cache,  process_mode

# %% [markdown]
# # Load MITgcm results

# %%
lake = 'geneva'
model = f'{lake}_2025'

# %%
print('Opening MITgcm dataset...')
mitgcm_config, ds_mitgcm = open_mitgcm_ds_from_config('../../config.json', model)
base_folder_path = os.path.dirname(mitgcm_config['datapath'])

# %%
horizontal_resolution = 100
ds_mitgcm['YG'] = np.arange(0, len(ds_mitgcm['YG'])) * horizontal_resolution
ds_mitgcm['XG'] = np.arange(0, len(ds_mitgcm['XG'])) * horizontal_resolution
ds_mitgcm['YC'] = np.arange(1, len(ds_mitgcm['YC']) + 1) * horizontal_resolution - horizontal_resolution / 2
ds_mitgcm['XC'] = np.arange(1, len(ds_mitgcm['XC']) + 1) * horizontal_resolution - horizontal_resolution / 2

# %% [markdown]
# # Get folder & files

# %%
base_folder = rf"/storage/alplakes_test/{lake}_100m_2025"

# %%
modal_analysis_dir = os.path.join(base_folder_path, "modal_analysis")

# %%
mode_date_str = "2025-08-07"
date_dir = Path(modal_analysis_dir) / mode_date_str
files = list(date_dir.rglob("*mode*.nc"))

# %% [markdown]
# # Process each layer

# %%
mode_cache = load_mode_cache(files)

# %%
import psutil

# %%
for month in range(4,12):
    for z_range in [range(0,10), range(10,20), range(20,30), range(30,40), range(40,50)]: #range(0,10), range(10,20), range(20,30), range(30,40), range(40,50)
        print(psutil.virtual_memory())
        print(f'Loading cropped dataset for month {month}...')
        ds_crop = ds_mitgcm.isel(Z=z_range).sel(
            time=slice(
                pd.to_datetime(f"2025-{month:02d}-01"),
                pd.to_datetime(f"2025-{month+1:02d}-01")
            )
        )

        ds_crop = ds_crop.load() #.chunk({'time': 100, 'Z': 1})

        print(f'Finished loading month {month}.')

        results = [
            process_mode(
                name,
                ds_mode,
                ds_crop.UVEL,
                ds_crop.VVEL,
                horizontal_resolution**2,
            )
            for name, ds_mode in mode_cache
        ]

        ds_modes = xr.concat(results, dim="mode").compute()
        
        print(f'Saving month {month:02d}, Z range {z_range[0]}-{z_range[-1]}...')
        ds_modes.to_netcdf(os.path.join(modal_analysis_dir, f"KE_month{month:02d}_Z{z_range[0]}-{z_range[-1]}.nc")) 
        print(f'Month {month}, Z range {z_range[0]}-{z_range[-1]} is done.')
        del ds_modes
        del ds_crop
        gc.collect()
