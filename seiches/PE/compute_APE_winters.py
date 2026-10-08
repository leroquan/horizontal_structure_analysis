"""
Equations and methods derived from:
>Winters K.B. (1995) _"Available potential energy and mixing in density-stratified fluids"_

Partitions the potential energy into background potential energy Eb and available potential energy Ea (sometimes noted APE in other publications).
***
**Background potential energy (Eb)** is definied as the minimum potential energy attainable through adiabatic (no change of temperature) redistribution of /rho into a reference state.

**Available potential energy (Ea/APE)** is the potential energy released in an adiabatic transition from the lake's state to the reference state.

**Total potential energy (Ep)** is the sum of background potential energy and available potential energy.

"""

import argparse
import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import sys
sys.path.append('../..//')
from utils_mitgcm import open_mitgcm_ds_from_config
import pylake

parser = argparse.ArgumentParser()
parser.add_argument("lake_name", type=str)
args = parser.parse_args()
lake = args.lake_name

g = 9.81

# ---------------------------------

print('Uploading MITgcm dataset...')
model = f'{lake}_2025'
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
ds['theta_nan'] = ds['THETA'].where(mask, np.nan)
rho = pylake.dens0(s=0.2, t=ds.theta_nan).astype(np.float64)

ref_z = ds.Zp1.values[-1]
z = ds.Z - ref_z
volume = (ds.drF * ds.rA).where(mask, np.nan).astype(np.float64)

# ---------------------------------

print('Precomputing time-independent quantities...')
volume_arr = np.asarray(volume, dtype=np.float64)
z_arr = np.asarray(z)

# Layers: bottom -> surface
layer_volume = volume.sum(dim=["XC", "YC"]).values
layer_z = z_arr.copy()

order_z = np.argsort(layer_z)
layer_volume = layer_volume[order_z]
layer_z = layer_z[order_z]

# Vertical coordinates for each cell
z_3d = np.broadcast_to(
    z_arr[:, None, None],
    volume_arr.shape
)

mask_flat = (
    ds.THETA.isel(time=0).values != 0
).ravel()

volume_flat = volume_arr.ravel()[mask_flat]
z_flat = z_3d.ravel()[mask_flat]

# ---------------------------------

print('Initializing results...')
nt = ds.sizes["time"]

Eb_all = np.full(nt, np.nan)
Ep_all = np.full(nt, np.nan)
Ea_all = np.full(nt, np.nan)

# ---------------------------------

print('Looping over time...')

for time_index in tqdm(range(nt)):


    rho_flat = np.asarray(
        rho.isel(time=time_index)
    ).ravel()[mask_flat]

    # Actual potential energy
    Ep = np.sum(g * rho_flat * volume_flat * z_flat)

    # Sort parcels: densest -> lightest
    idx_sort = np.argsort(rho_flat)[::-1]

    rho_sorted = rho_flat[idx_sort]
    vol_sorted = volume_flat[idx_sort]

    # Background potential energy
    Eb = 0.0

    parcel_idx = 0
    parcel_remaining = vol_sorted[0]

    for V_layer, z_layer in zip(layer_volume, layer_z):

        V_remaining = V_layer

        while V_remaining > 0 and parcel_idx < len(rho_sorted):

            V_used = min(V_remaining, parcel_remaining)

            Eb += (
                g
                * rho_sorted[parcel_idx]
                * V_used
                * z_layer
            )

            V_remaining -= V_used
            parcel_remaining -= V_used

            if parcel_remaining <= 1e-12:
                parcel_idx += 1

                if parcel_idx < len(rho_sorted):
                    parcel_remaining = vol_sorted[parcel_idx]

    # Store results
    Eb_all[time_index] = Eb
    Ep_all[time_index] = Ep
    Ea_all[time_index] = Ep - Eb

# ---------------------------------

print('Saving results...')
df_energy = pd.DataFrame({
    "time": ds.time.values,
    "Ep_[J]": Ep_all,
    "Eb_[J]": Eb_all,
    "Ea_[J]": Ep_all - Eb_all,
})

df_energy.reset_index().to_csv(os.path.join(output_folder, "EP_KBWinters1995.csv"))