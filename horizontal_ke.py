#%%
import pandas as pd
import os
#%%
folder_path = r"/storage/alplakes_test/lucerne_100m_2025"
input_folder = os.path.join(folder_path, "outputs_swirl", "eddy_catalogues_final")

output_folder = os.path.join(folder_path, "outputs_swirl", "ke_eddy")
os.makedirs(output_folder, exist_ok=True)
#%%
lvl0_csv_path = os.path.join(input_folder, "lvl0.csv")
lake_csv_path = os.path.join(input_folder, "lake_characteristics.csv")
#%%
df_lvl0 = pd.read_csv(lvl0_csv_path)
df_lvl0 = df_lvl0.set_index('id', drop=False)
df_lvl0['date'] = pd.to_datetime(df_lvl0['date'])
#%%
df_lake = pd.read_csv(lake_csv_path)
df_lake = df_lake.set_index('id', drop=False)
df_lake['date'] = pd.to_datetime(df_lake['date'])

#%%
def to_array_csvlike(x, dtype=float):
    if isinstance(x, str):
        return np.fromstring(x, sep=',', dtype=dtype)
    return np.asarray(x, dtype=dtype)
#%%
cols = ['i_eddy_cells', 'j_eddy_cells']
df_lvl0[cols] = df_lvl0[cols].map(lambda x: to_array_csvlike(x, dtype=np.int32))
#%%
import numpy as np
import sys
sys.path.append('..//')
from utils_mitgcm import open_mitgcm_ds_from_config
from utils_energy_analysis import *
#%%
model = 'lucerne_2025'
mitgcm_config, ds = open_mitgcm_ds_from_config('..//config.json', model)
#%%
grid_resolution = 100
ds['YC'] = np.arange(1, len(ds['YC']) + 1) * grid_resolution - grid_resolution / 2
ds['XC'] = np.arange(1, len(ds['XC']) + 1) * grid_resolution - grid_resolution / 2
ds['YG'] = np.arange(0, len(ds['YG'])) * grid_resolution
ds['XG'] = np.arange(0, len(ds['XG'])) * grid_resolution
#%%
mask = ds['THETA'].isel(time=0).values > 0
#%%
aligned_u = ds.UVEL.rename({'XG':'XC'})
aligned_u['XC'] = ds['XC']

aligned_v = ds.VVEL.rename({'YG':'YC'})
aligned_v['YC'] = ds['YC']

aligned_w = ds.WVEL.rename({'Zl':'Z'})
aligned_w['Z'] = ds['Z']
#%%
ke_tot = compute_ke(
    aligned_u,
    aligned_v,
    aligned_w,
    grid_resolution,
    grid_resolution,
    ds.drF)
#%%
def as_int(x):
    if isinstance(x, np.ndarray):
        return x.astype(int)
    return int(x)
#%%
ny = ke_tot.sizes["YC"]
nx = ke_tot.sizes["XC"]

EKE_map = np.zeros((ny, nx), dtype=np.float64)

for (t, z), grp in df_lvl0.groupby(["time_index", "depth_index"], sort=False):
    ke_slice = ke_tot.isel(time=int(t - 1), Z=int(z)).values  # (y, x)

    for row in grp.itertuples(index=False):
        j = as_int(row.j_eddy_cells)
        i = as_int(row.i_eddy_cells)
        EKE_map[j, i] += ke_slice[j, i]

#%%
np.save(os.path.join(output_folder, "EKE_map.npy"), EKE_map)