import xarray as xr
from scipy.signal import butter, filtfilt, detrend
import numpy as np


def check_filter_parameter_validity(btype, period_cutoff_low, period_cutoff_high):
    if not btype in ['bandpass', 'highpass', 'lowpass']:
        raise Exception('btype must be either "bandpass", "lowpass" or "highpass"')
    if btype == 'bandpass' and (period_cutoff_low is None or period_cutoff_high is None):
        raise Exception('Period cutoff low and period cutoff high must be specified for btype "bandpass"')
    if btype == 'lowpass' and period_cutoff_high is None:
        raise Exception('Period cutoff high must be specified for btype "lowpass"')
    if btype == 'highpass' and period_cutoff_low is None:
        raise Exception('Period cutoff low must be specified for btype "highpass"')


def filter_signal_timeseries(timeseries, btype, period_cutoff_low, period_cutoff_high, dt=1.0, order=5):
    """
    Apply band-pass Butterworth filter to a single time series using filtfilt.

    Parameters:
    -----------
    timeseries : array-like
        1D time series data
    btype: string
        'bandpass' or 'highpass' or 'lowpass'
    dt : float, optional
        Time step (sampling interval). Default is 1.0
    period_cutoff_low : float
        Lower cutoff period (longer period) in same units as dt.
    period_cutoff_high : float
        Upper cutoff period (shorter period) in same units as dt.
    order : int, optional
        Filter order. Default is 5

    Returns:
    --------
    filtered_ts : ndarray
        Band-pass filtered time series
    """

    # Remove NaNs with interpolation
    if np.any(np.isnan(timeseries)):
        valid_mask = ~np.isnan(timeseries)
        if np.sum(valid_mask) < len(timeseries) // 2:
            return np.full(len(timeseries), np.nan)
        timeseries = np.copy(timeseries)
        timeseries[~valid_mask] = np.interp(
            np.flatnonzero(~valid_mask),
            np.flatnonzero(valid_mask),
            timeseries[valid_mask]
        )

    timeseries = detrend(timeseries, type='linear')

    # Sampling frequency
    fs = 1.0 / dt
    nyq = fs / 2.0

    b,a = None,None
    if btype == 'bandpass':
        # Convert cutoff periods to frequencies
        fc_low = 1.0 / period_cutoff_low
        fc_high = 1.0 / period_cutoff_high

        # Normalised cutoff frequencies
        normal_cutoff = [fc_low / nyq, fc_high / nyq]

        if normal_cutoff[1] >= 1.0 or normal_cutoff[0] <= 0.0:
            return timeseries

        # Design bandpass filter
        b, a = butter(order, normal_cutoff, btype='band', analog=False)
    elif btype == 'highpass':
        # Convert cutoff periods to frequencies
        fc = 1.0 / period_cutoff_low  # Hz
        normal_cutoff = fc / nyq

        if normal_cutoff >= 1.0:
            return timeseries

        # Design filter
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
    elif btype == 'lowpass':
        # Convert cutoff periods to frequencies
        fc = 1.0 / period_cutoff_high  # Hz
        normal_cutoff = fc / nyq

        if normal_cutoff >= 1.0:
            return timeseries

        # Design **low-pass** filter
        b, a = butter(order, normal_cutoff, btype='low', analog=False)

    # Apply zero-phase filter
    filtered_ts = filtfilt(b, a, timeseries)

    return filtered_ts


def filter_signal_xarray(data, btype='bandpass', time_dim='time', period_cutoff_low=None, period_cutoff_high=None, dt=1.0, order=5):
    """
    Apply band-pass, low-pass or high-pass Butterworth filter over time dimension for each X,Y location in xarray.

    Parameters:
    -----------
    data : xarray.DataArray
        Input array with dimensions (T, Y, X) or any order containing T
    btype: string
        'bandpass' or 'highpass' or 'lowpass'
    time_dim : str, optional
        Name of time dimension. Default is 'T'
    dt : float, optional
        Time step (sampling interval). Default is 1.0
    period_cutoff_low : float
        Lower cutoff period (longer period edge of the band).
    period_cutoff_high : float
        Upper cutoff period (shorter period edge of the band).
    order : int, optional
        Filter order. Default is 5

    Returns:
    --------
    filtered_result : xarray.DataArray
        Band-pass filtered time series with same dimensions as input
    """
    check_filter_parameter_validity(btype, period_cutoff_low, period_cutoff_high)

    # Apply filter function using apply_ufunc
    filtered_result = xr.apply_ufunc(
        filter_signal_timeseries,
        data,
        btype,
        period_cutoff_low,
        period_cutoff_high,
        dt,
        order,
        input_core_dims=[[time_dim], [], [], [], [], []],
        output_core_dims=[[time_dim]],
        output_dtypes=[np.float64],
        dask='parallelized' if hasattr(data.data, 'chunks') else 'forbidden',
        vectorize=True,
        kwargs={}
    )



    # Preserve coordinates and attributes
    filtered_result = filtered_result.assign_coords(data.coords)

    # Add attributes

    str_btype=''
    if btype == 'bandpass':
        str_btype = 'Band-pass'
    elif btype == 'lowpass':
        str_btype = 'Low-pass'
    elif btype == 'highpass':
        str_btype = 'High-pass'

    filtered_result.attrs = data.attrs.copy()
    filtered_result.attrs['long_name'] = f'{str_btype} filtered {data.attrs.get("long_name", "data")}'
    filtered_result.attrs['filter_type'] = 'Butterworth {str_btype} (filtfilt)'
    filtered_result.attrs['cutoff_period_low'] = period_cutoff_low
    filtered_result.attrs['cutoff_period_high'] = period_cutoff_high
    filtered_result.attrs['filter_order'] = order
    filtered_result.attrs['units'] = data.attrs.get('units', 'units')

    return filtered_result

