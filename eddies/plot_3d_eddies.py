"""Interactive MITgcm lake, eddy-cell layers and horizontal currents.

Dependencies: numpy, pandas, matplotlib, plotly, MITgcmutils.
Coordinates remain in the model's Cartesian frame (not east/north).
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import socket
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from MITgcmutils import rdmds
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]


def snapshot_rows(path, timestamp, cache_dir):
    """Bounded-memory scan; cache invalidates when source size/mtime changes."""
    stat = path.stat()
    key = hashlib.sha256(f'{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{timestamp}'.encode()).hexdigest()[:20]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f'eddy_snapshot_{key}.csv'
    if cache.exists():
        return pd.read_csv(cache)
    selected = []
    columns = ['id', 'date', 'depth_index', 'depth_[m]', 'rotation_direction', 'i_eddy_cells', 'j_eddy_cells']
    for chunk in pd.read_csv(path, usecols=columns, chunksize=5000):
        match = chunk.loc[pd.to_datetime(chunk.date) == timestamp]
        if not match.empty:
            selected.append(match)
    result = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame(columns=columns)
    result.to_csv(cache, index=False)
    return result


def snapshot_rows_window(path, start, end, cache_dir):
    """Read and cache catalogue rows for a time window in one bounded scan."""
    stat = path.stat()
    key = hashlib.sha256(
        f'{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{start}:{end}'.encode()
    ).hexdigest()[:20]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f'eddy_window_{key}.csv'
    if cache.exists():
        return pd.read_csv(cache, parse_dates=['date'])
    selected = []
    columns = ['id', 'date', 'depth_index', 'depth_[m]', 'rotation_direction',
               'i_eddy_cells', 'j_eddy_cells']
    for chunk in pd.read_csv(path, usecols=columns, chunksize=100_000):
        dates = pd.to_datetime(chunk.date)
        match = dates.between(start, end)
        if match.any():
            rows = chunk.loc[match].copy()
            rows['date'] = dates.loc[match]
            selected.append(rows)
    result = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame(columns=columns)
    result.to_csv(cache, index=False)
    return result


def indices(value):
    # Both bracketed Python lists and comma-separated numbers occur in catalogues.
    parsed = np.atleast_1d(np.asarray(ast.literal_eval(str(value)), dtype=float))
    if not np.isfinite(parsed).all() or not np.equal(parsed, np.floor(parsed)).all():
        raise ValueError('Eddy cell indices must be finite integers')
    return parsed.astype(int)


def centered_velocity(u, v, wet_w, wet_s, wet_c):
    """Average bounding C-grid faces; closed/dry faces have zero flux.

    The absent outer east/north face is a closed boundary, never periodic.
    """
    u = np.where(wet_w > 0, u, 0.)
    v = np.where(wet_s > 0, v, 0.)
    east = np.zeros_like(u)
    north = np.zeros_like(v)
    east[..., :-1] = u[..., 1:]
    north[..., :-1, :] = v[..., 1:, :]
    return (np.where(wet_c > 0, (u + east) / 2, np.nan),
            np.where(wet_c > 0, (v + north) / 2, np.nan))


def _resolve_isotherm_h1(data, timestamp, h1):
    if h1 is None:
        modes_file = data.parent / 'modal_analysis' / 'modes_parameters.csv'
        if not modes_file.is_file():
            raise FileNotFoundError(f'No modal parameters file: {modes_file}')
        modes = pd.read_csv(modes_file, usecols=['date', 'h1'], parse_dates=['date'])
        if modes.empty:
            raise ValueError(f'No modal parameters in {modes_file}')
        nearest = int((modes['date'] - timestamp).abs().to_numpy().argmin())
        h1 = modes['h1'].to_numpy()[nearest]
    h1 = float(h1)
    if not np.isfinite(h1) or h1 <= 0:
        raise ValueError('Isotherm h1 must be a positive depth in metres')
    return h1


def _isotherm_height(theta, z, target, valid_columns):
    levels = np.asarray(z[:min(40, theta.shape[0])])
    if len(levels) < 2:
        raise ValueError('At least two temperature levels are required for an isotherm')
    diff = np.asarray(theta[:len(levels)] - target, dtype=float)
    diff[:, ~valid_columns] = np.nan
    valid = ~np.isnan(diff)
    last_valid = len(levels) - 1 - np.argmax(valid[::-1], axis=0)

    crossings = diff[:-1] * diff[1:] <= 0
    has_crossing = crossings.any(axis=0)
    idx = crossings.argmax(axis=0)
    d1 = np.take_along_axis(diff, idx[None, ...], axis=0)[0]
    d2 = np.take_along_axis(diff, (idx + 1)[None, ...], axis=0)[0]
    z1, z2 = levels[idx], levels[idx + 1]
    dz_over_dd = np.divide(z2 - z1, d2 - d1, out=np.zeros_like(d1), where=(d2 - d1) != 0)
    height = z1 - d1 * dz_over_dd

    no_crossing = ~has_crossing
    at_surface = no_crossing & (diff[0] <= 0)
    below_target = no_crossing & (diff[0] > 0)
    height[at_surface] = levels[0]
    height[below_target] = levels[last_valid[below_target]]
    height[~valid.any(axis=0)] = np.nan
    return height


def eddy_layer_quads(i, j, xe, ye, elevation, top, bottom, depth):
    """Center sheets plus exposed footprint walls over one model layer.

    Only horizontal boundary edges get walls: there are no internal cell walls.
    Bottoms are clipped to bathymetry, including partially wet bottom cells.
    """
    cells = set(zip(i, j))
    vertices = []
    for ii, jj in sorted(cells):
        x0, x1, y0, y1 = xe[ii], xe[ii+1], ye[jj], ye[jj+1]
        lower = max(bottom, -float(depth[jj, ii]))
        if lower >= top:
            continue
        center = max(elevation, lower)
        vertices.extend([(x0,y0,center),(x1,y0,center),
                         (x1,y1,center),(x0,y1,center)])
        edges = [((ii-1,jj), (x0,y0), (x0,y1)),
                 ((ii+1,jj), (x1,y1), (x1,y0)),
                 ((ii,jj-1), (x1,y0), (x0,y0)),
                 ((ii,jj+1), (x0,y1), (x1,y1))]
        for neighbor, a, b in edges:
            if neighbor not in cells:
                vertices.extend([(a[0],a[1],top), (b[0],b[1],top),
                                 (b[0],b[1],lower), (a[0],a[1],lower)])
    return vertices


def _add_energy_panel(fig, data_dir, timestamp, window_days):
    """Add the full-resolution lake KE curve and an exact-time animation marker."""
    path = data_dir.parent / 'outputs_swirl' / 'ke_eddy' / 'ke_lake.csv'
    energy = pd.read_csv(path, usecols=['date', 'kinetic_energy_[MJ]'], parse_dates=['date'])
    energy = energy.sort_values('date').set_index('date')['kinetic_energy_[MJ]']
    if energy.index.has_duplicates:
        raise ValueError(f'Duplicate kinetic-energy timestamps in {path}')
    half_window = pd.Timedelta(days=window_days / 2)
    start, end = timestamp - half_window, timestamp + half_window
    selected = energy.loc[start:end]
    if selected.empty:
        raise ValueError(f'No lake kinetic energy in the selected window: {path}')

    eddy_path = path.with_name('ke_eddy.csv')
    if not eddy_path.is_file():
        eddy_path = path.with_name('ke_eddies.csv')
    eddy_energy = pd.read_csv(eddy_path,
        usecols=['date', 'kinetic_energy_eddy_[MJ]'], parse_dates=['date'])
    eddy_energy = eddy_energy.sort_values('date').set_index('date')['kinetic_energy_eddy_[MJ]']
    if eddy_energy.index.has_duplicates:
        raise ValueError(f'Duplicate eddy kinetic-energy timestamps in {eddy_path}')
    eddy_selected = eddy_energy.loc[start:end]
    if eddy_selected.empty or not np.isfinite(eddy_selected.to_numpy()).any():
        raise ValueError(f'No finite eddy kinetic energy in the selected window: {eddy_path}')

    def marker(date):
        if date not in energy.index or not np.isfinite(energy.loc[date]):
            raise ValueError(f'No finite lake kinetic energy at exact snapshot time {date}: {path}')
        return go.Scatter(x=[date], y=[float(energy.loc[date])], mode='markers',
            marker=dict(color='red', size=11, line=dict(color='white', width=1)),
            name='Displayed date', showlegend=False, xaxis='x', yaxis='y',
            hovertemplate='%{x|%Y-%m-%d %H:%M}<br>Lake KE %{y:.2f} MJ<extra></extra>')

    fig.add_trace(go.Scatter(x=selected.index, y=selected.to_numpy(), mode='lines',
        line=dict(color='#294f73', width=2), name='Total lake kinetic energy',
        showlegend=False, xaxis='x', yaxis='y', connectgaps=False,
        hovertemplate='%{x|%Y-%m-%d %H:%M}<br>Lake KE %{y:.2f} MJ<extra></extra>'))
    fig.add_trace(go.Scatter(x=eddy_selected.index, y=eddy_selected.to_numpy(), mode='lines',
        line=dict(color='#d97706', width=2), name='Eddy kinetic energy',
        showlegend=False, xaxis='x', yaxis='y2', connectgaps=False,
        hovertemplate='%{x|%Y-%m-%d %H:%M}<br>Eddy KE %{y:.2f} MJ<extra></extra>'))
    marker_index = len(fig.data)
    fig.add_trace(marker(timestamp))
    for frame in fig.frames:
        frame.data = tuple(frame.data) + (marker(pd.Timestamp(frame.name)),)
        frame.traces = tuple(frame.traces) + (marker_index,)
    # Reserve extra height so the 3D scene remains large.
    fig.update_layout(height=(fig.layout.height or 1000) + 230,
        scene_domain=dict(x=[0, 1], y=[0, 0.70]),
        xaxis=dict(domain=[0, 1], anchor='y', type='date', title='Date',
                   range=[start, end] if start != end else
                         [start-pd.Timedelta(minutes=30), end+pd.Timedelta(minutes=30)]),
        yaxis=dict(domain=[0.81, 1], anchor='x',
                   title=dict(text='Total lake KE (MJ)', font=dict(color='#294f73')),
                   tickfont=dict(color='#294f73'), rangemode='tozero'),
        yaxis2=dict(overlaying='y', anchor='x', side='right',
                    title=dict(text='Eddy KE (MJ)', font=dict(color='#d97706')),
                    tickfont=dict(color='#d97706'), showgrid=False, rangemode='tozero'),
        legend=dict(y=0.69), margin=dict(l=85))
    # Keep 3D colorbars alongside the scene, below the energy panel.
    for trace in fig.data:
        if trace.type in ('surface', 'cone') and trace.showscale:
            trace.colorbar.y = (trace.colorbar.y or 0.5) * 0.70
            trace.colorbar.len = (trace.colorbar.len or 1) * 0.70
    if fig.layout.coloraxis.colorbar:
        fig.layout.coloraxis.colorbar.y = 0.78 * 0.70
        fig.layout.coloraxis.colorbar.len = 0.42 * 0.70
    for frame in fig.frames:
        for trace in frame.data:
            if trace.type in ('surface', 'cone') and trace.showscale:
                trace.colorbar.y = (trace.colorbar.y or 0.5) * 0.70
                trace.colorbar.len = (trace.colorbar.len or 1) * 0.70
        if frame.layout.coloraxis.colorbar:
            frame.layout.coloraxis.colorbar.y = 0.78 * 0.70
            frame.layout.coloraxis.colorbar.len = 0.42 * 0.70
    return fig


def build_figure(timestamp='2025-09-17 00:30:00', model='lucerne_2025',
                 catalogue=None, config_path=ROOT / 'config.json', host=None,
                 max_depth=80., arrow_depths=(2., 10., 30., 60.), stride=8,
                 exaggeration=35., arrow_seconds=2500., output_dir=None,
                 isotherm_h1=None, isotherm_stride=2, date_window_days=0,
                 date_step_hours=6, show_velocity=True, show_arrows=True,
                 show_eddies=True, show_thermocline=True, height=1000, width=None,
                 _catalogue_rows=None, _window_rows=None, _include_energy=True):
    """Return a Plotly figure. Time matching is exact; depths use nearest levels.

    Eddy layers retain their recorded footprints, with boundary walls spanning
    the corresponding MITgcm cell interfaces, clipped to the lakebed. Arrows show U/V only;
    their lengths equal horizontal speed times arrow_seconds, in model metres.
    """
    if (max_depth <= 0 or stride < 1 or exaggeration <= 0 or arrow_seconds <= 0
            or isotherm_stride < 1 or date_window_days < 0 or date_step_hours < 1):
        raise ValueError('Depth, stride, exaggeration and arrow_seconds must be positive')
    timestamp = pd.Timestamp(timestamp)
    cfg = json.loads(Path(config_path).read_text())[host or socket.gethostname()][model]
    data, grid = Path(cfg['datapath']), Path(cfg['gridpath'])
    if catalogue is None:
        # Same sibling directory layout used by plot_map_eddies.ipynb.
        catalogue = data.parent / 'outputs_swirl' / 'eddy_catalogues_final' / 'lvl0.csv'
    catalogue = Path(catalogue)
    output_dir = Path(output_dir or Path(__file__).parent / 'results_3d')
    window_rows = _window_rows
    if show_eddies and date_window_days and window_rows is None:
        half_window = pd.Timedelta(days=date_window_days / 2)
        window_rows = snapshot_rows_window(
            catalogue, timestamp - half_window, timestamp + half_window, output_dir / 'cache')
    steps = (timestamp - pd.Timestamp(cfg['ref_date'])).total_seconds() / cfg['dt']
    iteration = round(steps)
    if not np.isclose(steps, iteration, rtol=0, atol=1e-6):
        raise ValueError('Requested timestamp is not an exact MITgcm iteration')
    stem = data / f'3Dsnaps.{iteration:010d}'
    if not stem.with_suffix(stem.suffix + '.meta').exists():
        raise FileNotFoundError(f'No exact MITgcm snapshot for {timestamp}: {stem}.meta')
    fmt = cfg['endian']
    read = lambda name: rdmds(str(grid / name), machineformat=fmt)
    depth, xc, yc = read('Depth'), read('XC'), read('YC')
    z = read('RC').ravel()
    # Some land coordinates are zero; recover axes from populated grid rows/columns.
    x, y = xc.max(axis=0), yc.max(axis=1)
    if not (np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)):
        raise ValueError('This visualization requires a rectilinear Cartesian grid')
    lake = depth > 0
    if not (np.allclose(xc[lake], np.broadcast_to(x, depth.shape)[lake])
            and np.allclose(yc[lake], np.broadcast_to(y[:, None], depth.shape)[lake])):
        raise ValueError('Curvilinear grids are not supported')
    wet_c, wet_w, wet_s = read('hFacC'), read('hFacW'), read('hFacS')
    fields, _, meta = rdmds(str(stem), machineformat=fmt, returnmeta=True)
    names = [s.strip() for s in meta['fldlist']]
    if show_thermocline:
        if 'THETA' not in names:
            raise ValueError(f'MITgcm snapshot has no THETA field: {stem}')
        theta = fields[names.index('THETA')]
        isotherm_h1 = _resolve_isotherm_h1(data, timestamp, isotherm_h1)
        h1_level = int(np.argmin(np.abs(z + isotherm_h1)))
        if h1_level >= min(40, len(z)):
            raise ValueError('Isotherm h1 lies below the 40-level temperature search range')
        theta_h1 = theta[h1_level]
        valid_h1 = np.isfinite(theta_h1) & (theta_h1 != 0)
        if not valid_h1.any():
            raise ValueError('No valid temperatures at the selected isotherm h1')
        theta_h1_median = float(np.median(theta_h1[valid_h1]))
        isotherm_height = _isotherm_height(theta, z, theta_h1_median, valid_h1)
        isotherm_height = isotherm_height[::isotherm_stride, ::isotherm_stride]
        isotherm_depth = -isotherm_height
        lake_depth = depth[::isotherm_stride, ::isotherm_stride]
        isotherm_height = np.where(lake_depth > isotherm_depth, isotherm_height, np.nan)
        reference_height = float(np.nanmedian(isotherm_height))
        isotherm_displacement = isotherm_height - reference_height
        finite_displacement = isotherm_displacement[np.isfinite(isotherm_displacement)]
        if finite_displacement.size == 0:
            raise ValueError('No valid isotherm surface remains inside the lake bathymetry')
        displacement_limit = max(float(np.max(np.abs(finite_displacement))), 0.1)
    if show_eddies:
        if _catalogue_rows is not None:
            rows = _catalogue_rows.copy()
        elif window_rows is not None:
            rows = window_rows.loc[window_rows.date == timestamp].copy()
        else:
            rows = snapshot_rows(catalogue, timestamp, output_dir / 'cache')
        if rows.empty:
            raise ValueError(f'No eddies in catalogue at {timestamp}')
        rows = rows[rows['depth_[m]'].abs() <= max_depth]
        if rows.empty:
            raise ValueError('No eddy layers within the requested depth range')
        if not rows.rotation_direction.isin(['clockwise', 'anticlockwise']).all():
            raise ValueError('Unrecognized eddy rotation direction in catalogue')
    xx, yy = np.meshgrid(x / 1000, y / 1000)
    fig = go.Figure(go.Surface(x=xx, y=yy, z=np.where(depth > 0, -depth, np.nan),
        surfacecolor=depth, colorscale='Blues', opacity=0.35, showscale=False,
        name='Lakebed', hovertemplate='Lakebed: %{z:.1f} m<extra></extra>'))
    if show_thermocline:
        fig.add_trace(go.Surface(
            x=xx[::isotherm_stride, ::isotherm_stride],
            y=yy[::isotherm_stride, ::isotherm_stride], z=isotherm_height,
            surfacecolor=isotherm_displacement,
            colorscale=[[0, '#2166ac'], [0.5, '#f7f7f7'], [1, '#b2182b']],
            cmin=-displacement_limit, cmax=displacement_limit,
            colorbar=dict(title='Isotherm anomaly (m)', x=1.02, len=0.42, y=0.25,
                          tickvals=[-displacement_limit, 0, displacement_limit],
                          ticktext=[f'{-displacement_limit:.1f} sinking', 'reference',
                                    f'+{displacement_limit:.1f} rising']),
            opacity=0.9, showscale=True, connectgaps=False,
            name=f'Thermocline proxy ({theta_h1_median:.2f} °C)',
            legendgroup='thermocline', showlegend=True,
            customdata=isotherm_displacement,
            hovertemplate=f'Median isotherm: {theta_h1_median:.2f} deg C'
                          '<br>Elevation %{z:.1f} m'
                          '<br>Anomaly %{customdata:+.2f} m<extra></extra>'))
    ax = Figure().subplots()
    contours = ax.contour(xx, yy, depth > 0, levels=[.5])
    for line in contours.allsegs[0]:
        fig.add_trace(go.Scatter3d(x=line[:, 0], y=line[:, 1], z=np.zeros(len(line)),
            mode='lines', line=dict(color='#264653', width=4), name='Shoreline', showlegend=False))
    # Horizontal sheets and boundary walls preserve each catalogue footprint.
    xe = np.r_[x[0] - (x[1]-x[0])/2, (x[:-1]+x[1:])/2, x[-1]+(x[-1]-x[-2])/2] / 1000
    ye = np.r_[y[0] - (y[1]-y[0])/2, (y[:-1]+y[1:])/2, y[-1]+(y[-1]-y[-2])/2] / 1000
    excluded_cells = 0
    if show_eddies:
        z_faces = read('RF').ravel()
        if len(z_faces) != len(z) + 1 or not np.all(np.diff(z_faces) < 0):
            raise ValueError('Expected descending MITgcm vertical cell interfaces')
        for rotation, color in [('clockwise', '#F2D27A'), ('anticlockwise', '#8FCBB0')]:
            vertices, labels = [], []
            for _, row in rows[rows.rotation_direction == rotation].iterrows():
                i, j = indices(row.i_eddy_cells), indices(row.j_eddy_cells)
                k = int(row.depth_index)
                if len(i) != len(j) or np.any(i < 0) or np.any(i >= len(x)) or np.any(j < 0) or np.any(j >= len(y)):
                    raise ValueError(f'Invalid grid indices for eddy {row.id}')
                if k < 0 or k >= len(z) or not np.isclose(z[k], row['depth_[m]'], atol=.01):
                    raise ValueError('Catalogue and MITgcm depth grids differ')
                valid = wet_c[k, j, i] > 0
                excluded_cells += int((~valid).sum())
                i, j = i[valid], j[valid]
                layer_vertices = eddy_layer_quads(
                    i, j, xe, ye, z[k], z_faces[k], z_faces[k+1], depth)
                vertices.extend(layer_vertices)
                labels.extend([f'lvl0 id {row.id} | {rotation}'] * len(layer_vertices))
            verts = np.asarray(vertices, dtype=float).reshape((-1, 3))
            base = np.arange(0, len(verts), 4)
            fig.add_trace(go.Mesh3d(x=verts[:,0], y=verts[:,1], z=verts[:,2],
                i=np.r_[base,base], j=np.r_[base+1,base+2], k=np.r_[base+2,base+3],
                color=color, opacity=.65, flatshading=True,
                lighting=dict(ambient=0.65, diffuse=0.8, specular=0.15, roughness=0.8),
                name=f'Eddy layers: {rotation}', showlegend=True, legendgroup='eddies',
                text=labels, hovertemplate='%{text}<br>Depth %{z:.2f} m<extra></extra>'))
    if excluded_cells:
        warnings.warn(f'Clipped {excluded_cells} dry catalogue cells to the MITgcm wet mask', stacklevel=2)
    levels = sorted(set(int(np.argmin(abs(z + d))) for d in arrow_depths if 0 <= d <= max_depth))
    if show_velocity or show_arrows:
        u, v = centered_velocity(fields[names.index('UVEL')], fields[names.index('VVEL')], wet_w, wet_s, wet_c)
    if show_velocity:
        for n, k in enumerate(levels):
            speed_layer = np.hypot(u[k], v[k])
            fig.add_trace(go.Surface(
                x=xx, y=yy, z=np.where(np.isfinite(speed_layer), z[k], np.nan),
                surfacecolor=speed_layer, coloraxis='coloraxis', opacity=0.45,
                name=f'Velocity magnitude ({-z[k]:g} m)', legendgroup='velocity',
                showlegend=n == 0, connectgaps=False,
                customdata=speed_layer,
                hovertemplate='Speed %{customdata:.3f} m/s<br>Elevation %{z:.1f} m<extra></extra>'))
        speed_max = max((float(np.nanmax(np.hypot(u[k], v[k]))) for k in levels
                         if np.isfinite(u[k]).any()), default=0.1)
        fig.update_layout(coloraxis=dict(colorscale='Viridis', cmin=0, cmax=max(speed_max, 1e-9),
            colorbar=dict(title='Speed (m/s)', x=1.02, len=0.42, y=0.78)))
    if show_arrows:
        samples = []
        for k in levels:
            jj, ii = np.mgrid[0:len(y):stride, 0:len(x):stride]
            good = np.isfinite(u[k,jj,ii]) & np.isfinite(v[k,jj,ii])
            samples.extend(zip(ii[good], jj[good], np.full(good.sum(), k)))
        if samples:
            ii,jj,kk = np.asarray(samples).T
            uu,vv = u[kk,jj,ii],v[kk,jj,ii]
            speed = np.hypot(uu,vv)
            fig.add_trace(go.Cone(x=x[ii]/1000,y=y[jj]/1000,z=z[kk],
                u=uu*arrow_seconds/1000,v=vv*arrow_seconds/1000,w=np.zeros_like(uu),
                sizemode='raw', sizeref=1, anchor='tail', colorscale='Viridis',
                cmin=0, cmax=max((speed_max if show_velocity else speed.max())*arrow_seconds/1000,1e-9),
                showscale=not show_velocity, showlegend=True, legendgroup='arrows',
                colorbar=dict(title='Speed (m/s)', x=1.02, len=0.42, y=0.78, tickvals=np.linspace(0,speed.max()*arrow_seconds/1000,5),
                              ticktext=[f'{a:.3f}' for a in np.linspace(0,speed.max(),5)]),
                customdata=speed, name='Current arrows',
                hovertemplate='Speed %{customdata:.3f} m/s<br>Depth %{z:.1f} m<extra></extra>'))
        else:
            fig.add_trace(go.Cone(x=[], y=[], z=[], u=[], v=[], w=[],
                                  name='Current arrows', showlegend=False, showscale=False))
    fig.update_layout(height=height, width=width, autosize=width is None, uirevision='lake-view', title=f'{model} — {timestamp}<br><sup>Eddy layers to {max_depth:g} m; horizontal U/V; vertical exaggeration ×{exaggeration:g}; {excluded_cells} dry eddy cells clipped</sup>',
        scene=dict(xaxis_title='Model x (km)',yaxis_title='Model y (km)',zaxis_title='Elevation (m)',
                   aspectmode='manual',aspectratio=dict(x=1,y=np.ptp(y)/np.ptp(x),z=depth.max()*exaggeration/np.ptp(x)),
                   camera=dict(eye=dict(x=1.35,y=-1.6,z=1.1))),
        legend=dict(x=0,y=1, groupclick='togglegroup'), margin=dict(l=0,r=145,b=0,t=90))
    if date_window_days:
        fig = _add_date_slider(
            fig, timestamp, window_rows, cfg, model, catalogue, config_path, host,
            max_depth, arrow_depths, stride, exaggeration, arrow_seconds, output_dir,
            isotherm_h1, isotherm_stride, date_window_days, date_step_hours,
            dict(show_velocity=show_velocity, show_arrows=show_arrows,
                 show_eddies=show_eddies, show_thermocline=show_thermocline,
                 height=height, width=width))
    if _include_energy:
        fig = _add_energy_panel(fig, data, timestamp, date_window_days)
    return fig


def _add_date_slider(fig, timestamp, window_rows, cfg, model, catalogue, config_path,
                     host, max_depth, arrow_depths, stride, exaggeration,
                     arrow_seconds, output_dir, isotherm_h1, isotherm_stride,
                     date_window_days, date_step_hours, display_options):
    half_window = pd.Timedelta(days=date_window_days / 2)
    start, end = timestamp - half_window, timestamp + half_window
    desired_dates = list(pd.date_range(start, end, freq=pd.Timedelta(hours=date_step_hours)))
    if timestamp not in desired_dates:
        desired_dates.append(timestamp)
    available_dates = (set(pd.to_datetime(window_rows.date).unique())
                       if window_rows is not None else set(desired_dates))
    data = Path(cfg['datapath'])
    ref_date, dt = pd.Timestamp(cfg['ref_date']), cfg['dt']
    frame_dates = []
    for frame_date in sorted(set(desired_dates)):
        if frame_date not in available_dates:
            continue
        steps = (frame_date - ref_date).total_seconds() / dt
        iteration = round(steps)
        if not np.isclose(steps, iteration, rtol=0, atol=1e-6):
            continue
        stem = data / f'3Dsnaps.{iteration:010d}'
        if stem.with_suffix(stem.suffix + '.meta').exists():
            frame_dates.append(frame_date)
    if timestamp not in frame_dates:
        raise ValueError('The selected timestamp is unavailable in the date window')

    def trace_role(trace):
        if trace.name.startswith('Thermocline proxy'):
            return 'isotherm'
        if trace.name.startswith('Eddy layers:'):
            return trace.name
        if trace.name == 'Current arrows':
            return 'arrows'
        if trace.name.startswith('Velocity magnitude'):
            return trace.name
        return None

    base_roles = {trace_role(trace): index for index, trace in enumerate(fig.data)
                  if trace_role(trace) is not None}
    frames, slider_steps = [], []
    active = 0
    for frame_date in frame_dates:
        label = frame_date.strftime('%Y-%m-%d %H:%M')
        frame_rows = (window_rows.loc[pd.to_datetime(window_rows.date) == frame_date]
                      if window_rows is not None else None)
        if display_options['show_eddies'] and not (frame_rows['depth_[m]'].abs() <= max_depth).any():
            continue
        if frame_date == timestamp:
            frame_fig = fig
            active = len(frames)
        else:
            frame_fig = build_figure(
                timestamp=frame_date, model=model, catalogue=catalogue,
                config_path=config_path, host=host, max_depth=max_depth,
                arrow_depths=arrow_depths, stride=stride,
                exaggeration=exaggeration, arrow_seconds=arrow_seconds,
                output_dir=output_dir, isotherm_h1=isotherm_h1,
                isotherm_stride=isotherm_stride, _catalogue_rows=frame_rows,
                _window_rows=window_rows, _include_energy=False, **display_options)
        frame_roles = {trace_role(trace): trace for trace in frame_fig.data
                       if trace_role(trace) is not None}
        roles = list(base_roles)
        # Do not reset legend visibility when the date changes.
        frame_data = []
        for role in roles:
            payload = frame_roles[role].to_plotly_json()
            payload.pop('visible', None)
            frame_data.append(payload)
        frames.append(go.Frame(
            name=label,
            data=frame_data,
            traces=[base_roles[role] for role in roles],
            layout=go.Layout(title=frame_fig.layout.title.text, coloraxis=frame_fig.layout.coloraxis)))
        slider_steps.append(dict(
            method='animate', label=frame_date.strftime('%b %d %H:%M'),
            args=[[label], {'mode': 'immediate', 'frame': {'duration': 0, 'redraw': True},
                            'transition': {'duration': 0}}]))

    fig.frames = frames
    fig.update_layout(
        sliders=[dict(active=active, currentvalue=dict(prefix='Date: '),
                      pad=dict(t=30), steps=slider_steps)],
        updatemenus=[dict(type='buttons', direction='left', x=0, y=1.12,
                          showactive=False, buttons=[
            dict(label='Play', method='animate', args=[None, {
                'fromcurrent': True, 'mode': 'immediate',
                'frame': {'duration': 250, 'redraw': True},
                'transition': {'duration': 0}}]),
            dict(label='Pause', method='animate', args=[[None], {
                'mode': 'immediate', 'frame': {'duration': 0, 'redraw': False},
                'transition': {'duration': 0}}])])],
        margin=dict(l=0, r=145, b=95, t=110))
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--time', default='2025-09-17 00:30:00')
    parser.add_argument('--model', default='lucerne_2025')
    parser.add_argument('--catalogue', type=Path)
    parser.add_argument('--max-depth', type=float, default=80)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'results_3d/lake_eddies_3d.html')
    args = parser.parse_args()
    figure = build_figure(timestamp=args.time,model=args.model,catalogue=args.catalogue,max_depth=args.max_depth,output_dir=args.output.parent)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    figure.write_html(args.output,include_plotlyjs=True)
    print(args.output.resolve())
