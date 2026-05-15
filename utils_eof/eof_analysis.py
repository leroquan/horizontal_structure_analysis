import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


def rotary_eof_xarray(U, V, dA, n_modes=None):
    """
    Perform complex (rotary) EOF analysis on xarray Dataset with UVEL, VVEL.

    Parameters
    ----------
    U, V : xarray.Dataset
        Must have dims XC, YC, Z, time.
    dx, dy : float
        Horizontal grid spacing in meters (assumed uniform)
    rho : float or array-like, optional
        Density [kg/m³], either scalar or 1D array over z.
        Default: 1025 kg/m³ constant.
    n_modes : int, optional
        Number of EOF modes to retain. If None, keep all.

    Returns
    -------
    result : xarray.Dataset
        Contains complex EOFs (real=u pattern, imag=v pattern),
        principal components (PCs), modal KE, total KE, and explained variance.
    """

    # === 1. Align velocities ===
    dz = U.drF
    nt, nz, ny, nx = U.shape

    # === 2. Prepare weights ===
    rho_arr = xr.DataArray(
        np.full(dz.size, 1025.0),
        coords={"Z": dz["Z"]},
        dims=("Z",),
    )

    # Volume weight: sqrt(rho * dz * dA)
    w_z = np.sqrt(rho_arr * dz * dA)

    # complex weighted velocities
    Uw = (U * w_z).values.reshape(nt, nz*ny*nx)
    Vw = (V * w_z).values.reshape(nt, nz*ny*nx)
    Z = (Uw + 1j*Vw).T  # shape (nstate, nt)

    # === 3. Complex SVD ===
    Umat, S, VT = np.linalg.svd(Z, full_matrices=False)
    n_total = S.size
    if n_modes is None or n_modes > n_total:
        n_modes = n_total

    # truncate
    phi = Umat[:, :n_modes]
    PCs = (phi.conj().T @ Z).T  # (nt, n_modes)
    modal_ke_ts = 0.5 * np.abs(PCs) ** 2
    total_ke_ts = 0.5 * np.sum(np.abs(Z) ** 2, axis=0)

    # === 4. Map EOFs back to physical units ===
    w_vec = np.repeat(w_z.values, ny * nx)
    phi_phys = phi / w_vec[:, None]
    EOFs_phys = phi_phys.reshape(nz, ny, nx, n_modes).transpose(3, 0, 1, 2)

    # === 5. Compute explained variance fraction ===
    lam = 0.5 * S**2  # KE per mode
    frac = lam / lam.sum()

    # === 6. Package into xarray ===
    coords = {
        "mode": np.arange(1, n_modes + 1),
        "Z": U.Z,
        "YC": U.YC,
        "XC": U.XC,
        "time": U.time,
    }

    ds_out = xr.Dataset(
        {
            "EOF": (("mode", "Z", "YC", "XC"), EOFs_phys),
            "PC": (("time", "mode"), PCs),
            "modal_KE": (("time", "mode"), modal_ke_ts),
            "KE_total": ("time", total_ke_ts),
            "variance_fraction": ("mode", frac[:n_modes]),
        },
        coords=coords,
        attrs={
            "description": "Complex (rotary) EOF decomposition of UVEL + iVVEL",
            "weighting": "sqrt(rho * dz * dx * dy)",
            "units": {
                "EOF": "m/s (complex: real=u, imag=v)",
                "PC": "sqrt(J/kg)",
                "modal_KE": "J/kg",
            },
        },
    )

    return ds_out


def project_rotary_mode(U, V,  dA, eof_u, eof_v, rho=1025.0, normalize=True):
    """
    Project MITgcm velocity fields onto a prescribed Kelvin-wave mode
    and compute its time-dependent amplitude and kinetic energy.

    Parameters
    ----------


    eof_u, eof_v : xarray.DataArray
        Kelvin mode velocity structure (z, y, x) on tracer grid (XC, YC)

    rho : float or array-like
        Density [kg/m^3]; can be scalar or function of depth

    normalize : bool
        Normalize mode to unit energy norm

    Returns
    -------
    A : xarray.DataArray (time,) complex
        Kelvin wave amplitude

    KE : xarray.DataArray (time,)
        Kinetic energy of Kelvin mode
    """

    # --- 2. Align with EOF grid ---
    U, eof_u = xr.align(U, eof_u, join="exact")
    V, eof_v = xr.align(V, eof_v, join="exact")

    # --- 4. Build complex fields ---
    Z = U + 1j * V
    phi = eof_u + 1j * eof_v

    # --- 5. Build volume weights ---
    dz = U["drF"]

    rho_arr = xr.DataArray(
        np.full(dz.size, rho),
        coords={"Z": dz["Z"]},
        dims=("Z",),
    )

    # Volume weight: sqrt(rho * dz * dA)
    w = np.sqrt(rho_arr * dz * dA)

    # --- 6. Apply weights ---
    Zw = (Z * w).stack(space=("Z", "YC", "XC"))
    phiw = (phi * w).stack(space=("Z", "YC", "XC"))

    # Convert to numpy for fast linear algebra
    Zw_np = Zw.values
    phiw_np = phiw.values

    # --- 7. Normalize mode ---
    if normalize:
        norm = np.sqrt(np.vdot(phiw_np, phiw_np).real)
        if norm == 0:
            raise ValueError("Mode has zero norm.")
        phiw_np = phiw_np / norm

    # --- 8. Projection (complex amplitude) ---
    A = Zw_np @ phiw_np.conj()

    # --- 9. Kinetic energy ---
    KE = 0.5 * np.abs(A) ** 2

    # --- 10. Wrap outputs ---
    A_da = xr.DataArray(A, coords={"time": U["time"]}, dims=("time",), name="A_kelvin")
    KE_da = xr.DataArray(KE, coords={"time": U["time"]}, dims=("time",), name="KE_kelvin")

    return A_da, KE_da


def print_and_save_figure_rotary_eof(mode, rotary, output_folder, date_str, subsetting_factor=5, figsize=(15, 4), zz=0):
    EOF = rotary.EOF.isel(mode=mode)

    # U and V components
    U_pattern = EOF.real
    V_pattern = EOF.imag

    # Compute horizontal amplitude
    amp = np.sqrt(U_pattern ** 2 + V_pattern ** 2)

    # Plot quiver for a single depth slice
    plt.figure(figsize=figsize)
    amp.isel(Z=zz).plot(add_colorbar=False)
    plt.quiver(rotary.XC[::subsetting_factor], rotary.YC[::subsetting_factor],
               U_pattern[zz, :, :][::subsetting_factor, ::subsetting_factor],
               V_pattern[zz, :, :][::subsetting_factor, ::subsetting_factor],
               scale=1e-5)
    plt.gca().invert_yaxis()
    plt.title(f'Rotary EOF mode {mode + 1}, depth level {zz}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(label='Amplitude')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{date_str}_eof_mode{mode}_depth{zz}_pattern.png"))

    PC = rotary.PC.isel(mode=mode)
    plt.figure(figsize=figsize)
    plt.plot(rotary.time, PC, label='Amplitude')
    plt.ylabel('Amplitude (sqrt(KE))')
    plt.xlabel('Time')
    plt.title(f'PC amplitude, mode {mode + 1}')
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{date_str}_eof_mode{mode}_depth{zz}_PC.png"))

    plt.figure(figsize=(8,4))
    rotary.variance_fraction.plot(marker='o', markersize=5)
    plt.ylabel('Fraction of variance explained [-]')
    plt.xlabel('Mode')
    plt.title(f'Fraction of variance explained')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{date_str}_eof_variance.png"))


