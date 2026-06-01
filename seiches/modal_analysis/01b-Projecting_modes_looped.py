# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
import sys

from partd import python

sys.path.append(os.path.abspath('../../'))
from utils_mitgcm import *
from utils_modal_analysis import _match_mode_to_field, project_vector_mode
# %% [markdown]
# # Load MITgcm results
# %%
lake = 'neuchatel'
model = f'{lake}_2025'
# %%
print(fr"Loading MITgcm dataset for {model}...")
mitgcm_config, ds_mitgcm = open_mitgcm_ds_from_config('../../config.json', model)
base_folder_path = os.path.dirname(mitgcm_config['datapath'])
# %%
horizontal_resolution = 100
ds_mitgcm['YG'] = np.arange(0, len(ds_mitgcm['YG'])) * horizontal_resolution
ds_mitgcm['XG'] = np.arange(0, len(ds_mitgcm['XG'])) * horizontal_resolution
ds_mitgcm['YC'] = np.arange(1, len(ds_mitgcm['YC']) + 1) * horizontal_resolution - horizontal_resolution / 2
ds_mitgcm['XC'] = np.arange(1, len(ds_mitgcm['XC']) + 1) * horizontal_resolution - horizontal_resolution / 2
# %% [markdown]
# # Get dates to loop on
# %%
modal_analysis_dir = os.path.join(base_folder_path, "modal_analysis")
# %%
# Extract and list the dates of the extracted EOF patterns
_all_eof_subfolders = [
    name for name in os.listdir(modal_analysis_dir)
    if os.path.isdir(os.path.join(modal_analysis_dir, name))
]

_pat = re.compile(r"(\d{4}-\d{2}-\d{2})$")

mode_folders = [name for name in _all_eof_subfolders if _pat.match(name)]
mode_dates_str = [m.group(1) for name in mode_folders for m in [_pat.match(name)]]
mode_dates = pd.to_datetime(mode_dates_str)
# %% [markdown]
# # Process each date
# %%
from pathlib import Path
# %%
ds_all = []

print(fr"Looping on each date and projecting modes...")
for idx_date in range(len(mode_dates)):
    mode_date = mode_dates[idx_date] # loop here on each date
    ds_crop = ds_mitgcm.sel(time=slice(pd.to_datetime("2025-04-01"), pd.to_datetime("2025-12-01"))) #.sel(time=slice(mode_date - pd.Timedelta(days=7), mode_date + pd.Timedelta(days=7)))

    date_dir = Path(modal_analysis_dir) / mode_dates_str[idx_date]
    files = list(date_dir.rglob("*mode*.nc"))

    for nc_file in files:
        ds_mode = xr.open_dataset(nc_file)
        h1_mode = ds_mode.h1.values

        ds_mode['u1'] = (ds_mode.u1_real + 1j * ds_mode.u1_imag).fillna(0)
        ds_mode['v1'] = (ds_mode.v1_real + 1j * ds_mode.v1_imag).fillna(0)

        mode_u1 = ds_mode.u1.expand_dims(Z=ds_crop.Z) #.where(ds_crop.Z >= -1*h1_mode, drop=True)
        mode_v1 = ds_mode.v1.expand_dims(Z=ds_crop.Z) #.where(ds_crop.Z >= -1*h1_mode, drop=True)

        mode_u = mode_u1 # could also concatenate mode_u1 and mode_u2
        mode_v = mode_v1

        U = ds_crop.UVEL
        V = ds_crop.VVEL
        dA = horizontal_resolution^2

        proj_each_slice = []
        KE_arr = []
        A_real_arr = []
        A_imag_arr = []
        for idx_Z in range(50,len(ds_crop.Z)):
            print(fr'Computing layer {idx_Z}')
            tmp_results = project_vector_mode(U.isel(Z=idx_Z), V.isel(Z=idx_Z), dA, mode_u.isel(Z=idx_Z), mode_v.isel(Z=idx_Z), rho=1025.0)
            proj_each_slice.append(tmp_results)
            KE_arr.append(tmp_results['KE'].expand_dims(Z=[ds_crop.Z.isel(Z=idx_Z)]))
            A_real_arr.append(tmp_results['A_real'].expand_dims(Z=[ds_crop.Z.isel(Z=idx_Z)]))
            A_imag_arr.append(tmp_results['A_imag'].expand_dims(Z=[ds_crop.Z.isel(Z=idx_Z)]))

        xr_KE = xr.concat(KE_arr, dim="Z").expand_dims(mode=[str(nc_file)]).expand_dims(mode_date=[mode_date])
        xr_A_real = xr.concat(A_real_arr, dim="Z").expand_dims(mode=[str(nc_file)]).expand_dims(mode_date=[mode_date])
        xr_A_imag = xr.concat(A_imag_arr, dim="Z").expand_dims(mode=[str(nc_file)]).expand_dims(mode_date=[mode_date])

        ds_result = xr.Dataset({"KE":xr_KE, "A_real":xr_A_real, "A_imag": xr_A_imag})
        ds_all.append(ds_result)
# %%
print(fr"Concatenating results and saving to netcdf...")
ds_results = xr.concat(ds_all, dim="mode")
ds_results.to_netcdf(os.path.join(modal_analysis_dir, "KE_projected_50-80.nc"))
