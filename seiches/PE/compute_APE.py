import sys
import os
from multiprocessing.pool import ExceptionWithTraceback

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from numpy.f2py.auxfuncs import throw_error

sys.path.append('../..//')
from utils_mitgcm import open_mitgcm_ds_from_config
import pylake

model = 'neuchatel_2025'

mitgcm_config, ds = open_mitgcm_ds_from_config('../../config.json', model)
folder_path = os.path.dirname(mitgcm_config['datapath'])
output_folder = os.path.join(folder_path, "seiche_analysis", "potential_energy")
os.makedirs(output_folder, exist_ok=True)

grid_resolution = 100
ds['YC'] = np.arange(1, len(ds['YC'])+1) * grid_resolution - grid_resolution/2
ds['XC'] = np.arange(1, len(ds['XC'])+1) * grid_resolution - grid_resolution/2
ds['YG'] = np.arange(0, len(ds['YG'])) * grid_resolution
ds['XG'] = np.arange(0, len(ds['XG'])) * grid_resolution
mask = ds.THETA.isel(time=0).values != 0
plt.figure(figsize=(10, 3))
zz = 25
tt = 24 * 8 - 10
plt.imshow(ds.THETA.isel(time=tt, Z=zz).where(mask[zz], np.nan))
plt.text(0.02, 0.98, f'Z={ds.Z.isel(Z=zz).values:.2f}', ha='left', va='top', transform=plt.gca().transAxes)
t = ds.time.isel(time=tt).values
plt.title(np.datetime_as_string(t, unit='m').replace('T', ' '))
plt.gca().invert_yaxis()
plt.colorbar()

z_index= 15
xc_index = 100
yc_index = 40

ds['THETA'].isel(XC=xc_index,YC=yc_index).plot(y='Z')
plt.gca().invert_yaxis()
z_index= 15
xc_index = 100
yc_index = 40

ds['THETA'].isel(YC=yc_index, time=tt).plot(y='Z')
plt.gca().invert_yaxis()
# Get density
g = 9.81
ds['theta_nan'] = ds['THETA'].where(mask, np.nan)
rho = pylake.dens0(s=0.2, t=ds.theta_nan).astype(np.float64)
# Get Total Potential Energy Epot
ref_z = ds.Zp1.values[-1]
z = ds.Z - ref_z
volume = (ds.drF * ds.rA).where(mask, np.nan).astype(np.float64)
Epot = g * rho * z * volume.astype(np.float64)
Epot_sum = Epot.sum(dim=['XC', 'YC', 'Z'])
df_Epot = Epot.sum(dim=['XC', 'YC', 'Z']).to_dataframe(name='Epot_tot')['Epot_tot']
df_Epot.reset_index().to_csv(os.path.join(output_folder, "EPp_KBWinters1995.csv"))
df_Epot.plot()
# Get Background Potential Energy
def volume_to_depth(V_query, V_cum_lake, z_grid):
    """
    Map cumulative volume to depth using bathymetry.

    You must precompute:
        V(z) from bathymetry
    then invert it.
    """

    return np.interp(V_query, V_cum_lake, z_grid)
def compute_background_potential_energy(time_index, rho, volume_flat, z_flat, mask_flat):

    v_cum_lake = np.cumsum(volume_flat[mask_flat])
    rho_arr = np.asarray(rho.isel(time=time_index)).ravel()

    rho_arr = rho_arr[mask_flat]
    # -------------------------
    # Sort densities (reference state)
    # -------------------------
    idx = np.argsort(rho_arr) #sorted from higher to lower
    rho_sorted = rho_arr[idx]
    V_sorted = volume_flat[mask_flat][idx]
    # -------------------------
    # Build sorted vertical positions
    # -------------------------
    # cumulative volume coordinate
    V_cum = np.cumsum(V_sorted)

    # assign depths assuming monotonic mapping z(V)
    # (you must provide or precompute from bathymetry)
    z_sorted = volume_to_depth(V_cum, v_cum_lake, z_flat[mask_flat])

    # -------------------------
    # 4. Sorted potential energy
    # -------------------------
    E_background = np.sum(rho_sorted * g * z_sorted * V_sorted)

    return E_background
time_index = 24 * 7 - 10
volume_flat = np.asarray(volume).astype(np.float64).ravel()
z_flat = np.repeat(np.asarray(z), volume.sizes["YC"] * volume.sizes["XC"])
mask_flat = mask.ravel()
Eb_snap = compute_background_potential_energy(time_index, rho, volume_flat, z_flat, mask_flat)
df_Epot.iloc[time_index] - Eb_snap
Eb = []
for i in range(len(ds.time)):
    Eb_temp = compute_background_potential_energy(i, rho, volume_flat, z_flat, mask_flat)
    Eb.append(Eb_temp)
if isinstance(df_Epot, pd.Series):
    df_Epot = df_Epot.to_frame(name="Epot_tot")

df_Epot["Eb"] = pd.Series(Eb, index=df_Epot.index)

df_Epot['APE'] = df_Epot['Epot_tot'] - df_Epot['Eb']
df_Epot['APE'].plot()
df_Epot.reset_index().to_csv(os.path.join(output_folder, "EP_KBWinters1995.csv"))
