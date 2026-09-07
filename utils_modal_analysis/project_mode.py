import numpy as np
import xarray as xr


def _match_mode_to_field(field, mode):
    """
    Put a 2D mode field onto the horizontal grid of a model velocity field.

    This handles common MITgcm staggered-grid issues where the mode may contain
    one extra XG/YG point compared with the velocity output, where coordinate
    labels differ slightly, and where interpolation onto the field coordinates
    is needed.
    """
    mode = mode.squeeze(drop=True)

    # Keep only horizontal dimensions that are present in both objects.
    common_dims = [dim for dim in mode.dims if dim in field.dims]

    # First trim by size when the mode has one extra staggered-grid point and
    # the coordinates already coincide with the beginning of the field grid.
    indexers = {}
    for dim in common_dims:
        if mode.sizes[dim] == field.sizes[dim] + 1:
            indexers[dim] = slice(0, field.sizes[dim])
    if indexers:
        mode = mode.isel(indexers)

    # Interpolate the mode onto the field coordinates whenever coordinates or
    # sizes do not already match. Values outside the mode domain are set to 0.
    interp_coords = {}
    assign_coords = {}
    for dim in common_dims:
        if dim in mode.coords and dim in field.coords:
            same_size = mode.sizes[dim] == field.sizes[dim]
            same_coords = same_size and np.array_equal(mode[dim].values, field[dim].values)

            if same_coords:
                assign_coords[dim] = field[dim]
            else:
                interp_coords[dim] = field[dim]

    if interp_coords:
        mode = mode.interp(interp_coords, kwargs={"fill_value": 0}).fillna(0)

    if assign_coords:
        mode = mode.assign_coords(assign_coords)

    # Final strict check after intentional grid matching.
    _, mode = xr.align(field.isel({dim: 0 for dim in field.dims if dim not in mode.dims}), mode, join="exact")

    return mode


def load_mode_cache(files):
    mode_cache = []
    for nc_file in files:
        ds_mode = xr.open_dataset(nc_file).load()

        ds_mode["u1"] = (
            ds_mode.u1_real +
            1j * ds_mode.u1_imag
        ).fillna(0)

        ds_mode["v1"] = (
            ds_mode.v1_real +
            1j * ds_mode.v1_imag
        ).fillna(0)

        ds_mode["u2"] = (
            ds_mode.u2_real +
            1j * ds_mode.u2_imag
        ).fillna(0)

        ds_mode["v2"] = (
            ds_mode.v2_real +
            1j * ds_mode.v2_imag
        ).fillna(0)

        mode_cache.append(
            (
                str(nc_file),
                ds_mode
            )
        )

    return mode_cache


def prepare_vector_mode(
    U,
    V,
    mode_u,
    mode_v,
    dA,
    rho=1025.0,
    normalize_mode=True,
):
    """
    Precompute all mode-dependent quantities, including Z-dependent weights.
    """

    mode_u = _match_mode_to_field(U.isel(Z=0), mode_u)
    mode_v = _match_mode_to_field(V.isel(Z=0), mode_v)

    # Z-dependent weights
    w_u = np.sqrt(rho * U["drF"] * dA)
    w_v = np.sqrt(rho * V["drF"] * dA)

    # Weighted modes
    mode_uw = mode_u * w_u
    mode_vw = mode_v * w_v

    mode_uw_np = []
    mode_vw_np = []
    norm2 = []
    
    def get_norm2(mode_uw_np_i, mode_vw_np_i):
        # Mode norm
        norm2_i = (
            np.vdot(mode_uw_np_i, mode_uw_np_i).real
            +
            np.vdot(mode_vw_np_i, mode_vw_np_i).real
        )

        return norm2_i

    for zz in range(len(w_u)):
        mode_uw_np_i = mode_uw.isel(Z=zz).values.ravel()
        mode_uw_np.append(mode_uw_np_i)

        mode_vw_np_i = mode_vw.isel(Z=zz).values.ravel()
        mode_vw_np.append(mode_vw_np_i)

        norm2.append(
            get_norm2(
                mode_uw_np_i, 
                mode_vw_np_i))

    norm = np.sqrt(norm2)

    if normalize_mode:

        norm = np.sqrt(norm2)

        mode_u = mode_u.expand_dims({"Z": U["Z"]}) / xr.DataArray(norm, dims=["Z"])
        mode_v = mode_v.expand_dims({"Z": U["Z"]}) / xr.DataArray(norm, dims=["Z"])

        mode_uw_np /= norm[:, None]
        mode_vw_np /= norm[:, None]

        norm2 = np.ones_like(norm2)

    return {
        "mode_u": mode_u,
        "mode_v": mode_v,

        "mode_uw_np": mode_uw_np,
        "mode_vw_np": mode_vw_np,

        "norm2": norm2,

        "w_u": w_u,
        "w_v": w_v,
    }


def project_vector_mode_level(
    U,
    V,
    prepared_mode,
    zz
):
    """
    Project one vertical level onto a precomputed vector mode.
    """


    mode_uw_np = prepared_mode["mode_uw_np"][zz]
    mode_vw_np = prepared_mode["mode_vw_np"][zz]

    w_u = prepared_mode["w_u"][zz]
    w_v = prepared_mode["w_v"][zz]

    norm2 = prepared_mode["norm2"][zz]

    mode_u = prepared_mode["mode_u"][zz]
    mode_v = prepared_mode["mode_v"][zz]


    # ------------------------------------------------------------
    # Weighted velocity
    # ------------------------------------------------------------

    Uw_np = (
        (U * w_u)
        .values
        .reshape(U.sizes["time"], -1)
    )

    Vw_np = (
        (V * w_v)
        .values
        .reshape(V.sizes["time"], -1)
    )


    # ------------------------------------------------------------
    # Projection amplitude
    # ------------------------------------------------------------

    A = (
        Uw_np @ mode_uw_np.conj()
        +
        Vw_np @ mode_vw_np.conj()
    )


    # ------------------------------------------------------------
    # Kinetic energy
    # ------------------------------------------------------------

    KE = 0.5 * np.abs(A)**2 / norm2


    # ------------------------------------------------------------
    # Reconstruction
    # ------------------------------------------------------------

    # U_proj_complex_np = (
    #     A[:, None, None]
    #     *
    #     mode_u.values[None, :, :]
    #     /
    #     norm2
    # )

    # V_proj_complex_np = (
    #     A[:, None, None]
    #     *
    #     mode_v.values[None, :, :]
    #     /
    #     norm2
    # )


    # U_proj_complex = xr.DataArray(
    #     U_proj_complex_np,
    #     coords={
    #         "time": U.time,
    #         "YC": U.YC,
    #         "XG": U.XG,
    #     },
    #     dims=("time", "YC", "XG"),
    # )


    # V_proj_complex = xr.DataArray(
    #     V_proj_complex_np,
    #     coords={
    #         "time": V.time,
    #         "YG": V.YG,
    #         "XC": V.XC,
    #     },
    #     dims=("time", "YG", "XC"),
    # )


    A_da = xr.DataArray(
        A,
        coords={"time": U.time},
        dims="time",
    )


    KE_da = xr.DataArray(
        KE,
        coords={"time": U.time},
        dims="time",
    )


    return {
        "A_real": A_da.real,
        "A_imag": A_da.imag,
        "KE": KE_da,

        #"U_real": U_proj_complex.real,
        #"U_imag": U_proj_complex.imag,

        #"V_real": V_proj_complex.real,
        #"V_imag": V_proj_complex.imag,
    }


def process_mode(
    mode_name,
    ds_mode,
    U,
    V,
    dA
):

    print(f'Processing mode {mode_name}...')
    prepared_mode = prepare_vector_mode(
        U,
        V,
        ds_mode.u1,
        ds_mode.v1,
        dA=dA,
        rho=1025.0,
    )


    results = []

    for zz in range(U.sizes["Z"]):

        result = project_vector_mode_level(
            U.isel(Z=zz),
            V.isel(Z=zz),
            prepared_mode,
            zz
        )

        ds = xr.Dataset(result)

        ds = ds.expand_dims(
            Z=[U.Z.values[zz]]
        )

        results.append(ds)


    return (
        xr.concat(results, dim="Z")
        .expand_dims(mode=[mode_name])
    )


def project_vector_mode(
    U,
    V,
    dA,
    mode_u,
    mode_v,
    rho=1025.0,
    normalize_mode=True,
):
    """
    KE-weighted projection of MITgcm vector velocity fields onto
    a possibly complex vector mode.

    Parameters
    ----------
    U, V : xarray.DataArray
        Velocity fields on MITgcm C-grid:
            U : (time, Z, YC, XG)
            V : (time, Z, YG, XC)

    dA : xarray.DataArray
        Horizontal cell area.

    mode_u, mode_v : xarray.DataArray
        Complex modal structure on the same grids as U and V.

    rho : float
        Reference density.

    normalize_mode : bool
        If True, normalize mode so that ||mode||² = 1
        under the KE inner product.

    Returns
    -------
    A_da : xr.DataArray
        Complex modal amplitude A(t)

    KE_da : xr.DataArray
        Modal kinetic energy:
            KE = 0.5 * |A|²
        if normalize_mode=True

    U_proj_phys, V_proj_phys : xr.DataArray
        Physical reconstructed velocities:
            Re(A * mode)

    U_proj_complex, V_proj_complex : xr.DataArray
        Full complex reconstructed fields.
    """

    # ------------------------------------------------------------
    # Match mode grids to velocity grids
    # ------------------------------------------------------------

    mode_u = _match_mode_to_field(U, mode_u)
    mode_v = _match_mode_to_field(V, mode_v)

    has_z = "Z" in U.dims
    if has_z:
        rho_u_arr = xr.DataArray(
            np.full(U["drF"].size, rho),
            coords={"Z": U["drF"]["Z"]},
            dims=("Z",),
        )
        rho_v_arr = xr.DataArray(
            np.full(V["drF"].size, rho),
            coords={"Z": V["drF"]["Z"]},
            dims=("Z",),
        )
    else:
        rho_u_arr = rho
        rho_v_arr = rho

    # ------------------------------------------------------------
    # KE weighting
    #
    # Inner product:
    #
    # <u,v> = ∫ rho * u * v* dV
    #
    # We implement this via sqrt(weights)
    # ------------------------------------------------------------

    w_u = np.sqrt(rho_u_arr * U["drF"] * dA)
    w_v = np.sqrt(rho_v_arr * V["drF"] * dA)

    # ------------------------------------------------------------
    # Weighted fields
    # ------------------------------------------------------------

    if has_z:
        u_space = ("Z", "YC", "XG")
        v_space = ("Z", "YG", "XC")
    else:
        u_space = ("YC", "XG")
        v_space = ("YG", "XC")

    Uw = (U * w_u).stack(space=u_space)
    Vw = (V * w_v).stack(space=v_space)

    mode_uw = (mode_u * w_u).stack(space=u_space)
    mode_vw = (mode_v * w_v).stack(space=v_space)

    Uw_np = Uw.values
    Vw_np = Vw.values

    mode_uw_np = mode_uw.values
    mode_vw_np = mode_vw.values

    # ------------------------------------------------------------
    # Mode norm
    # ------------------------------------------------------------

    norm2 = (
        np.vdot(mode_uw_np, mode_uw_np).real
        + np.vdot(mode_vw_np, mode_vw_np).real
    )

    if normalize_mode:
        norm = np.sqrt(norm2)

        mode_u = mode_u / norm
        mode_v = mode_v / norm

        mode_uw_np = mode_uw_np / norm
        mode_vw_np = mode_vw_np / norm

        norm2 = 1.0

    # ------------------------------------------------------------
    # Modal amplitude
    #
    # A(t) = <u, mode>
    # ------------------------------------------------------------

    A = (
        Uw_np @ mode_uw_np.conj()
        + Vw_np @ mode_vw_np.conj()
    )

    # ------------------------------------------------------------
    # Modal KE
    #
    # If normalized:
    #     KE = 0.5 |A|²
    # ------------------------------------------------------------

    KE = 0.5 * np.abs(A) ** 2 / norm2

    # ------------------------------------------------------------
    # Complex reconstruction
    #
    # u_proj = A * mode / ||mode||²
    # ------------------------------------------------------------

    A_expanded = A[:, None]

    mode_u_stack = mode_u.stack(space=u_space).values
    mode_v_stack = mode_v.stack(space=v_space).values

    U_proj_complex_np = (
        A_expanded * mode_u_stack / norm2
    )

    V_proj_complex_np = (
        A_expanded * mode_v_stack / norm2
    )

    # ------------------------------------------------------------
    # Convert back to xarray
    # ------------------------------------------------------------

    U_proj_complex = xr.DataArray(
        U_proj_complex_np.reshape(Uw.shape),
        coords=Uw.coords,
        dims=Uw.dims,
    ).unstack("space")

    V_proj_complex = xr.DataArray(
        V_proj_complex_np.reshape(Vw.shape),
        coords=Vw.coords,
        dims=Vw.dims,
    ).unstack("space")


    # ------------------------------------------------------------
    # Output DataArrays
    # ------------------------------------------------------------

    A_da = xr.DataArray(
        A,
        coords={"time": U["time"]},
        dims=("time",),
        name="A_mode",
    )

    KE_da = xr.DataArray(
        KE,
        coords={"time": U["time"]},
        dims=("time",),
        name="KE_mode",
    )

    return {
        'A_real': A_da.real,
        'A_imag': A_da.imag,
        'KE': KE_da,
        'U_real': U_proj_complex.real,
        'V_real': V_proj_complex.real,
        'U_imag': U_proj_complex.imag,
        'V_imag': V_proj_complex.imag,
    }