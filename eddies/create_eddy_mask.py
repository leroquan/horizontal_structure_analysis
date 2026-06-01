#%%
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import xarray as xr
#%%
folder_path = r"/storage/alplakes_test/lucerne_100m_2025"
input_folder = os.path.join(folder_path, "outputs_swirl", "eddy_catalogues_final")

output_folder = os.path.join(folder_path, "outputs_swirl", "ke_eddy")
os.makedirs(output_folder, exist_ok=True)
#%%
lvl0_csv_path = os.path.join(input_folder, "lvl0.csv")
lake_csv_path = os.path.join(input_folder, "lake_characteristics.csv")
#%% md
# # Get datasets
#%%
df_lvl0 = pd.read_csv(lvl0_csv_path)
df_lvl0 = df_lvl0.set_index('id', drop=False)
df_lvl0['date'] = pd.to_datetime(df_lvl0['date'])
#%%
ds_ke = xr.open_dataarray(r"/storage/alplakes_test/lucerne_100m_2025/energy_budget/kinetic_energy.nc")
#%% md
# # Get eddy indices
#%%
def to_array_csvlike(x, dtype=float):
    if isinstance(x, str):
        return np.fromstring(x, sep=',', dtype=dtype)
    return np.asarray(x, dtype=int)
#%%
df = df_lvl0.copy()
df["i_list"] = df["i_eddy_cells"].apply(to_array_csvlike)
df["j_list"] = df["j_eddy_cells"].apply(to_array_csvlike)
#%%
df_compact = df[["time_index", "depth_index", "eddy_index", "i_list", "j_list"]].copy()
df_compact['time_index'] = df['time_index'] - 1
df_compact.to_parquet(r"/storage/alplakes_test/lucerne_100m_2025/outputs_swirl/eddy_catalogues_final/eddy_cells.parquet", index=False)
#%%
