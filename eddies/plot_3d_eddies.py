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


def build_figure(timestamp='2025-09-17 00:30:00', model='lucerne_2025',
                 catalogue=None, config_path=ROOT / 'config.json', host=None,
                 max_depth=80., arrow_depths=(2., 10., 30., 60.), stride=8,
                 exaggeration=35., arrow_seconds=2500., output_dir=None,
                 isotherm_h1=None, isotherm_stride=2):
    """Return a Plotly figure. Time matching is exact; depths use nearest levels.

    Eddy layers use every occupied lvl0 cell at its recorded depth. They do not
    imply a reconstructed boundary between sampled depths. Arrows show U/V only;
    their lengths equal horizontal speed times arrow_seconds, in model metres.
    """
    if (max_depth <= 0 or stride < 1 or exaggeration <= 0 or arrow_seconds <= 0
            or isotherm_stride < 1):
        raise ValueError('Depth, stride, exaggeration and arrow_seconds must be positive')
    timestamp = pd.Timestamp(timestamp)
    cfg = json.loads(Path(config_path).read_text())[host or socket.gethostname()][model]
    data, grid = Path(cfg['datapath']), Path(cfg['gridpath'])
    if catalogue is None:
        # Same sibling directory layout used by plot_map_eddies.ipynb.
        catalogue = data.parent / 'outputs_swirl' / 'eddy_catalogues_final' / 'lvl0.csv'
    catalogue = Path(catalogue)
    output_dir = Path(output_dir or Path(__file__).parent / 'results_3d')
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
    if 'THETA' not in names:
        raise ValueError(f'MITgcm snapshot has no THETA field: {stem}')
    theta = fields[names.index('THETA')]
    u, v = centered_velocity(fields[names.index('UVEL')], fields[names.index('VVEL')], wet_w, wet_s, wet_c)
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
    fig.add_trace(go.Surface(
        x=xx[::isotherm_stride, ::isotherm_stride],
        y=yy[::isotherm_stride, ::isotherm_stride], z=isotherm_height,
        surfacecolor=isotherm_displacement,
        colorscale=[[0, '#b2182b'], [0.5, '#f7f7f7'], [1, '#2166ac']],
        cmin=-displacement_limit, cmax=displacement_limit,
        colorbar=dict(title='Isotherm elevation anomaly (m)',
                      tickvals=[-displacement_limit, 0, displacement_limit],
                      ticktext=[f'{-displacement_limit:.1f} sinking', 'reference',
                                f'+{displacement_limit:.1f} rising']),
        opacity=0.9, showscale=True, connectgaps=False,
        name=f'Median isotherm ({theta_h1_median:.2f} deg C)',
        customdata=isotherm_displacement,
        hovertemplate=f'Median isotherm: {theta_h1_median:.2f} deg C'
                      '<br>Elevation %{z:.1f} m'
                      '<br>Anomaly %{customdata:+.2f} m<extra></extra>'))
    ax = Figure().subplots()
    contours = ax.contour(xx, yy, depth > 0, levels=[.5])
    for line in contours.allsegs[0]:
        fig.add_trace(go.Scatter3d(x=line[:, 0], y=line[:, 1], z=np.zeros(len(line)),
            mode='lines', line=dict(color='#264653', width=4), name='Shoreline', showlegend=False))
    # Two triangles per occupied cell preserve the catalogue footprint exactly.
    xe = np.r_[x[0] - (x[1]-x[0])/2, (x[:-1]+x[1:])/2, x[-1]+(x[-1]-x[-2])/2] / 1000
    ye = np.r_[y[0] - (y[1]-y[0])/2, (y[:-1]+y[1:])/2, y[-1]+(y[-1]-y[-2])/2] / 1000
    excluded_cells = 0
    for rotation, color in [('clockwise', '#E69F00'), ('anticlockwise', '#009E73')]:
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
            for ii, jj in zip(i, j):
                vertices.extend([(xe[ii],ye[jj],z[k]),(xe[ii+1],ye[jj],z[k]),
                                 (xe[ii+1],ye[jj+1],z[k]),(xe[ii],ye[jj+1],z[k])])
                labels.extend([f'lvl0 id {row.id} | {rotation}'] * 4)
        if vertices:
            verts = np.asarray(vertices)
            base = np.arange(0, len(verts), 4)
            fig.add_trace(go.Mesh3d(x=verts[:,0], y=verts[:,1], z=verts[:,2],
                i=np.r_[base,base], j=np.r_[base+1,base+2], k=np.r_[base+2,base+3],
                color=color, opacity=.32, name=f'Eddy layers: {rotation}', showlegend=True,
                text=labels, hovertemplate='%{text}<br>Depth %{z:.2f} m<extra></extra>'))
    if excluded_cells:
        warnings.warn(f'Clipped {excluded_cells} dry catalogue cells to the MITgcm wet mask', stacklevel=2)
    levels = sorted(set(int(np.argmin(abs(z + d))) for d in arrow_depths if 0 <= d <= max_depth))
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
            cmin=0, cmax=max(speed.max()*arrow_seconds/1000,1e-9),
            colorbar=dict(title='Speed (m/s)', tickvals=np.linspace(0,speed.max()*arrow_seconds/1000,5),
                          ticktext=[f'{a:.3f}' for a in np.linspace(0,speed.max(),5)]),
            customdata=speed, name='Horizontal velocity',
            hovertemplate='Speed %{customdata:.3f} m/s<br>Depth %{z:.1f} m<extra></extra>'))
    fig.update_layout(title=f'{model} — {timestamp}<br><sup>Eddy layers to {max_depth:g} m; horizontal U/V; vertical exaggeration ×{exaggeration:g}; {excluded_cells} dry eddy cells clipped</sup>',
        scene=dict(xaxis_title='Model x (km)',yaxis_title='Model y (km)',zaxis_title='Elevation (m)',
                   aspectmode='manual',aspectratio=dict(x=1,y=np.ptp(y)/np.ptp(x),z=depth.max()*exaggeration/np.ptp(x)),
                   camera=dict(eye=dict(x=1.35,y=-1.6,z=1.1))),
        legend=dict(x=0,y=1), margin=dict(l=0,r=0,b=0,t=75))
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
