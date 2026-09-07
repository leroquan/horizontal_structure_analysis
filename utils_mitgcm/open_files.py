import xmitgcm as xm
import json
import numpy as np
import socket


def convert_ds_to_little_endian(mitgcm_ds):
    mitgcm_ds = mitgcm_ds.astype('<f8')
    mitgcm_ds= mitgcm_ds.assign(Z=mitgcm_ds['Z'].astype('<f8'))
    mitgcm_ds= mitgcm_ds.assign(XC=mitgcm_ds['XC'].astype('<f8'))
    mitgcm_ds= mitgcm_ds.assign(YC=mitgcm_ds['YC'].astype('<f8'))
    mitgcm_ds= mitgcm_ds.assign(XG=mitgcm_ds['XG'].astype('<f8'))
    mitgcm_ds= mitgcm_ds.assign(YG=mitgcm_ds['YG'].astype('<f8'))
    mitgcm_ds= mitgcm_ds.assign(Zp1=mitgcm_ds['Zp1'].astype('<f8'))
    mitgcm_ds= mitgcm_ds.assign(Zu=mitgcm_ds['Zu'].astype('<f8'))
    mitgcm_ds= mitgcm_ds.assign(Zl=mitgcm_ds['Zl'].astype('<f8'))
    
    return mitgcm_ds


def open_mitgcm_ds(datapath, gridpath, ref_date, dt_mitgcm_results, endian):
    ds = xm.open_mdsdataset(
                            datapath, 
                            grid_dir=gridpath, 
                            ref_date=ref_date, 
                            prefix='3Dsnaps', 
                            delta_t=dt_mitgcm_results, 
                            endian=endian)
    if endian == '>':
        ds = convert_ds_to_little_endian(ds)

    return ds


def open_mitgcm_ds_from_config(config_path, model):
    with open(config_path, 'r') as file:
        mitgcm_config = json.load(file)[socket.gethostname()][model]
        
    datapath = mitgcm_config['datapath']
    gridpath = mitgcm_config['gridpath']
    ref_date = mitgcm_config['ref_date']
    dt_mitgcm_results = mitgcm_config['dt']
    endian = mitgcm_config['endian']

    return mitgcm_config, open_mitgcm_ds(datapath, gridpath, ref_date, dt_mitgcm_results, endian)


def align_coordinates(ds):
    horizontal_resolution = 100
    ds['YG'] = np.arange(0, len(ds['YG'])) * horizontal_resolution
    ds['XG'] = np.arange(0, len(ds['XG'])) * horizontal_resolution
    ds['YC'] = np.arange(1, len(ds['YC'])+1) * horizontal_resolution - horizontal_resolution/2
    ds['XC'] = np.arange(1, len(ds['XC'])+1) * horizontal_resolution - horizontal_resolution/2

    aligned_u = ds.UVEL.rename({'XG':'XC'})
    aligned_u['XC'] = ds['XC']

    aligned_v = ds.VVEL.rename({'YG':'YC'})
    aligned_v['YC'] = ds['YC']

    aligned_w = ds.WVEL.rename({'Zl':'Z'})
    aligned_w['Z'] = ds['Z']

    return ds, aligned_u, aligned_v, aligned_w


def repair_coord(ds, grid_resolution):
    ds['YC'] = np.arange(1, len(ds['YC'])+1) * grid_resolution - grid_resolution/2
    ds['XC'] = np.arange(1, len(ds['XC'])+1) * grid_resolution - grid_resolution/2
    ds['YG'] = np.arange(0, len(ds['YG'])) * grid_resolution
    ds['XG'] = np.arange(0, len(ds['XG'])) * grid_resolution

    return ds