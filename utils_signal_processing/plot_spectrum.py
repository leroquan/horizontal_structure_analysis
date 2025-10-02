import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter

from utils_signal_processing import compute_fft_period

# Custom formatter for tick labels with 2 significant figures
def custom_formatter(x, pos):
    return f'{x:.1f}'


def plot_freq_spectrum(xr_fft, var_name: str, depth: float, m_segm: float, y_lim_min=1e-9, x_lim_min=0.01e-4, fontsize=16):
    fig, ax = plt.subplots(1, figsize=(15, 6))
    # Plot the mean amplitude spectrum
    xr_fft.amp_mean.plot(ax=ax)

    # Add uncertainty shading
    ax.fill_between(xr_fft['freq'],
                    xr_fft.err_low,
                    xr_fft.err_up,
                    color='gray', alpha=0.5, label='Uncertainty')

    # Set both axes to log scale
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Grid and labels
    ax.grid()
    ax.set_xlabel('Frequency $(s^{-1})$', fontsize=fontsize)
    ax.set_ylabel('Power Spectrum', fontsize=fontsize)

    # Increase the number of ticks using LogLocator
    ax.xaxis.set_major_locator(LogLocator(base=10, subs='auto', numticks=8))

    # Create a secondary x-axis for period in hours
    secax = ax.secondary_xaxis('top')
    secax.set_xscale('log')
    secax.set_xlabel('Period (hours)', fontsize=fontsize)

    # Compute FFT period ticks
    fft_ticks = ax.get_xticks()
    fft_period = compute_fft_period(fft_ticks)  # In seconds
    fft_hr = fft_period / 3600

    secax.set_xticks(fft_ticks)
    secax.set_xticklabels([f'{int(x):d}' for x in fft_hr], fontsize=fontsize)  # Set tick labels without decimals and in non-scientific notation form and increase tick label size

    ax.xaxis.set_major_formatter(FuncFormatter(custom_formatter)) # Set primary x-axis tick labels with 2 significant figures
    ax.xaxis.get_offset_text().set_visible(False)  # Manually set the offset text for the power limits and hide the default offset text
    x_min, x_max = ax.get_xlim()
    power_offset = int(np.floor(np.log10(x_max))) # Compute the offset manually

    # Set the custom offset text
    ax.annotate(f'$\\times 10^{power_offset}$', xy=(1, 0), xycoords='axes fraction',
                    fontsize=fontsize, xytext=(-30, -30), textcoords='offset points',
                    ha='center', va='center')

    # Adjust tick labels by dividing by the power offset
    def adjusted_formatter(x, pos):
        return f'{x / 10**power_offset:.1f}'

    ax.xaxis.set_major_formatter(FuncFormatter(adjusted_formatter))

    # Set axis limits
    ax.set_ylim(y_lim_min, None)
    ax.set_xlim(x_lim_min, None)

    # Increase the size of the primary and secondary tick labels
    plt.setp(ax.get_xticklabels(), fontsize=fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize)
    plt.setp(secax.get_xticklabels(), fontsize=fontsize)

    plt.title('{} - Depth:{}m - Segments for FFT:{}'.format(var_name,depth,m_segm), fontsize=fontsize+2)

    return fig,ax