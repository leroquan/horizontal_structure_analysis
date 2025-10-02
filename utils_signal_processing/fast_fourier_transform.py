import numpy as np
import xarray as xr

from scipy.stats.distributions import chi2
from scipy.signal import butter, filtfilt, detrend

def compute_mean_fft(data, M):
    """Computes mean FFT by segmenting the time series."""
    N = len(data)
    p = N // M
    if p < 2:
        raise ValueError(f"Segment length (p={p}) is too small for FFT. Choose a smaller M.")

    data = data[:p * M]
    data_segments = data.reshape(M, p)
    window = np.hanning(p) * np.ones([M, 1])

    segments_demean = np.array([seg - np.nanmean(seg) for seg in data_segments])
    segments_demean[np.isnan(segments_demean)] = 0
    data_dtrend = detrend(segments_demean, axis=1, type='linear')
    fft_segments = np.fft.fft(data_dtrend * window, axis=1)
    amp_segments = abs(fft_segments[:, :p // 2] / p) ** 2
    amp_segments[:, 1:] *= 2
    amp_mean = amp_segments.mean(axis=0)

    nu = 2 * M
    err_up = nu / chi2.ppf(0.1 / 2, df=nu) * amp_mean
    err_low = nu / chi2.ppf(1 - 0.1 / 2, df=nu) * amp_mean

    return amp_mean, err_up, err_low


def xr_compute_meanfft(data, M):
    """Parallelized FFT computation for (X, Y) grid along T."""

    dt = (data['time'][1] - data['time'][0]).astype('float').values / 1e9
    N = len(data['time'].values)
    p = N // M
    freq = np.fft.fftfreq(p, dt)[:p // 2]
    len_freq = len(freq)

    amp_mean, err_up, err_low = xr.apply_ufunc(
        compute_mean_fft,
        data,
        M,
        input_core_dims=[["time"], []],
        output_core_dims=[["freq"], ["freq"], ["freq"]],
        output_sizes={"freq": len_freq},
        exclude_dims=set(("time",)),
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64, np.float64, np.float64],
    )

    fft_data = xr.Dataset(
        {
            "amp_mean": amp_mean,
            "err_up": err_up,
            "err_low": err_low,
        },
        coords={"freq": freq}
    )

    return fft_data


def compute_fft_period(fft_freq):
    # Time period
    with np.errstate(divide='ignore', invalid='ignore'):
        fft_period = np.where(fft_freq != 0, 1 / fft_freq, np.inf)

    return fft_period