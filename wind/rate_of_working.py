from pandas import date_range

"""
Based on the paper of Simpson & Woolway (2021)
https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2020WR029441
"""
import pandas as pd
import numpy as np
import xarray as xr
import os
import matplotlib.pyplot as plt

import sys
sys.path.append('..//')
from utils_mitgcm import open_mitgcm_ds_from_config
from utils_energy_analysis import *

lake = 'zug'
base_folder = rf"/storage/alplakes_test/{lake}_100m_2025"
output_folder = os.path.join(base_folder, 'wind_analysis')
os.makedirs(output_folder, exist_ok=True)
bin_folder = os.path.join(rf'/home/leroquan@eawag.wroot.emp-eaw.ch/work_space/{lake}_100m_2025', 'binary_data')
suffix=''

def save_as_compressed_netcdf(ds: xr.DataArray, name: str, path: str):
    ds.name = name

    encoding = {
        name: {
            "zlib": True,
            "complevel": 4,     # 1–9, 4 is a good default
        }
    }

    ds.to_netcdf(path, encoding=encoding)

# Import datasets
ds_wind = xr.open_dataset(os.path.join(bin_folder, "wind.nc"))
ds_wind['time'] = ds_wind.time + np.timedelta64(30, 'm')

model = f'{lake}_2025'
mitgcm_config, ds = open_mitgcm_ds_from_config('..//config.json', model)
grid_resolution = 100
ds['YC'] = np.arange(1, len(ds['YC']) + 1) * grid_resolution - grid_resolution / 2
ds['XC'] = np.arange(1, len(ds['XC']) + 1) * grid_resolution - grid_resolution / 2
ds['YG'] = np.arange(0, len(ds['YG'])) * grid_resolution
ds['XG'] = np.arange(0, len(ds['XG'])) * grid_resolution
mask = ds['THETA'].isel(time=0).values > 0

tau_x = xr.open_dataarray(os.path.join(output_folder, f'wind_tau_x{suffix}.nc'))
tau_y = xr.open_dataarray(os.path.join(output_folder, f'wind_tau_y{suffix}.nc'))

# Wind Rate of Working
ds_surface = ds.isel(Z=0)

u = ds_surface.UVEL.rename({'XG':'x', 'YC':'y'})
u['x'] = tau_x['x']
u['y'] = tau_y['y']
v = ds_surface.VVEL.rename({'XC':'x', 'YG':'y'})
v['y'] = tau_y['y']
v['x'] = tau_x['x']
tau_x_crop = tau_x.sel(time=slice(u.time.values[0],u.time.values[-1]+np.timedelta64(2, 'm')))
tau_y_crop = tau_y.sel(time=slice(u.time.values[0],u.time.values[-1]+np.timedelta64(2, 'm')))
u['time'] = tau_x_crop.time
v['time'] = tau_x_crop.time
RW = tau_x * u + tau_y * v # kg/s2/m * m/s = kg/s3 = m2/m2 * kg/s3 = kg*m2/s3 /m2 = W/m2
save_as_compressed_netcdf(RW, 'RW', os.path.join(output_folder, f'wind_RW_Wperm2{suffix}.nc'))

RW = xr.open_dataarray(os.path.join(output_folder, f'wind_RW_Wperm2{suffix}.nc')) # W/m2
RW_mean = RW.mean(dim=['x', 'y'])
RW_mean.to_netcdf(os.path.join(base_folder, 'wind_analysis', f'mean_wind_RW_Wperm2{suffix}.nc'))


nb_cells = RW.isel(time=0).count()
cell_area = grid_resolution**2
total_area = nb_cells * cell_area

E_input_wind = RW_mean * 3600 * total_area/1e6 # W/m2=J/s/m2 --> MJ/h
E_input_wind.name = "E_wind_MJperh"
E_input_wind.to_netcdf(os.path.join(base_folder, 'wind_analysis', f'E_wind_MJperh{suffix}.nc'))
E_input_wind.to_dataframe()['E_wind_MJperh'].reset_index().to_csv(os.path.join(base_folder, 'wind_analysis', f'E_wind_MJperh{suffix}.csv'))