import numpy as np
import xarray as xr

from scipy.stats.distributions import chi2
from scipy.signal import detrend


def compute_mean_fft(data, seg_length):
    """Computes mean FFT by segmenting the time series.

    Parameters
    ----------
    data : 1D array-like
        Time series.
    seg_length : int
        Segment length (samples per segment).

    Returns
    -------
    amp_mean, err_up, err_low : 1D np.ndarray
        Mean power spectrum estimate and chi-square confidence bounds.
        Arrays have length seg_length//2.
    """
    data = np.asarray(data)
    N = data.size
    seg_length = int(seg_length)

    if seg_length < 2:
        raise ValueError(f"Segment length (seg_length={seg_length}) must be >= 2.")

    # Number of full segments available
    M = N // seg_length
    if M < 1:
        raise ValueError(
            f"Not enough samples for seg_length={seg_length}. "
            f"Need at least {seg_length} samples, got N={N}. "
            f"Choose a smaller seg_length or provide a longer time series."
        )

    # Trim to a whole number of segments and reshape
    data = data[: M * seg_length]
    data_segments = data.reshape(M, seg_length)

    # Window for each segment
    window = np.hanning(seg_length)[None, :]  # shape (1, seg_length), broadcasts over M

    # Demean each segment (nan-safe); replace remaining NaNs with 0 for detrend/FFT
    seg_means = np.nanmean(data_segments, axis=1, keepdims=True)
    segments_demean = data_segments - seg_means
    segments_demean = np.where(np.isnan(segments_demean), 0.0, segments_demean)

    # Detrend each segment
    data_dtrend = detrend(segments_demean, axis=1, type="linear")

    # FFT and single-sided power spectrum
    fft_segments = np.fft.fft(data_dtrend * window, axis=1)
    amp_segments = (np.abs(fft_segments[:, : seg_length // 2]) / seg_length) ** 2
    if seg_length // 2 > 1:
        amp_segments[:, 1:] *= 2.0  # conserve power for single-sided spectrum (excluding DC)

    amp_mean = amp_segments.mean(axis=0)

    # Confidence bounds (chi-square), degrees of freedom ~ 2*M (Welch average)
    nu = 2 * M
    alpha = 0.1  # 90% CI
    err_up = (nu / chi2.ppf(alpha / 2, df=nu)) * amp_mean
    err_low = (nu / chi2.ppf(1 - alpha / 2, df=nu)) * amp_mean

    return amp_mean, err_up, err_low


def xr_compute_meanfft(data, seg_length):
    """Parallelized FFT computation for (X, Y) grid along time."""
    dt = (data["time"][1] - data["time"][0]).astype("float").values / 1e9

    freq = np.fft.fftfreq(seg_length, dt)[: seg_length // 2]
    len_freq = len(freq)

    amp_mean, err_up, err_low = xr.apply_ufunc(
        compute_mean_fft,
        data,
        seg_length,
        input_core_dims=[["time"], []],
        output_core_dims=[["freq"], ["freq"], ["freq"]],
        output_sizes={"freq": len_freq},
        exclude_dims=set(("time",)),
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64, np.float64, np.float64],
    )

    fft_data = xr.Dataset(
        {"amp_mean": amp_mean, "err_up": err_up, "err_low": err_low},
        coords={"freq": freq},
    )
    return fft_data


def compute_fft_period(fft_freq):
    """Convert FFT frequency array to period (1/f), with inf at f=0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(fft_freq != 0, 1 / fft_freq, np.inf)


from scipy.signal.windows import dpss
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatter
import numpy as np
def rotary_psd(u, v, f, dt=1.0, NW=3, K=5, plot=True):
    """
    Compute rotary PSD for 2D vector time series (u, v) with multitaper.
    Works for odd or even length signals.
    """
    z = np.asarray(u) + 1j*np.asarray(v)
    n = len(z)

    # Generate DPSS tapers
    tapers = dpss(n, NW, Kmax=K)

    # FFT length
    nfft = n
    pos_len = nfft // 2 if n % 2 == 0 else nfft // 2 + 1  # number of positive freqs

    # Initialize spectra
    Spp = np.zeros(pos_len)
    Snn = np.zeros(pos_len)

    # Loop over tapers
    for k in range(K):
        tapered = z * tapers[k]
        Z = np.fft.fft(tapered)/n
        Spp += np.abs(Z[:pos_len])**2
        Snn += np.abs(Z[-pos_len:][::-1])**2  # flip negative freqs to match pos axis

    Spp /= K
    Snn /= K

    # Frequency axis (positive)
    freqs_pos = np.fft.fftfreq(n, dt)[:pos_len]

    # Plot
    if plot:
        fig, ax = plt.subplots(1,1, figsize=(8,5))

        ax.loglog(freqs_pos, Spp, label='Positive (CW)')
        ax.loglog(freqs_pos, Snn, label='Negative (CCW)')

        ax.vlines(f, 0, Spp.max(), linestyle='--', color='k', label='Coriolis Frequency')

        ax.xaxis.set_major_locator(LogLocator(base=10, subs=np.arange(1,10,3)))

        # Create a secondary x-axis for period in hours
        secax = ax.secondary_xaxis('top')
        secax.set_xscale('log')
        secax.set_xlabel('Period (hours)')
        secax.set_xticks(ax.get_xticks())
        secax.set_xticklabels([f'{int(x):d}' for x in compute_fft_period(ax.get_xticks()) / 3600])


        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [unit^2/Hz]')
        plt.title('Rotary Power Spectral Density')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    return freqs_pos, Spp, Snn