# Interactive lake and eddy figure

Open `plot_3d_eddies.ipynb` and run its cells. The default is Lucerne at
2025-09-17 00:30:00, matching a snapshot in `plot_map_eddies.ipynb`.
Change `timestamp`, `model`, `max_depth`, `arrow_depths`, `stride`,
`exaggeration`, or `arrow_seconds` in the settings cell.

The notebook exports a standalone HTML file in `results_3d/`, which can be
opened offline in a browser. Rotate/zoom with the mouse and toggle eddy
layers through the legend. Velocity colors and hover values are in m/s.
Arrow length equals horizontal speed multiplied by `arrow_seconds`;
it is an instantaneous velocity glyph, not a particle trajectory.

The figure uses:

- MITgcm `Depth` for the translucent lakebed and surface shoreline.
- The actual `i_eddy_cells` / `j_eddy_cells` from `lvl0.csv`, rendered as
  pale, translucent cell sheets at each detection depth (gold clockwise,
  mint anticlockwise). These are sampled layers, not interpolated eddy volumes;
  their subdued colors keep the red/blue isotherm easier to read.
- `UVEL` / `VVEL` from the exact matching `3Dsnaps` output, averaged from
  their bounding staggered faces onto cell centers. Dry faces are closed.
  Vertical velocity is not shown. Arrows use the nearest model levels to
  requested depths; hover reports the actual depth.
- `THETA` from the same snapshot as a 3D median-temperature surface. The
  target is the horizontal median at the nearest model level to `h1`, with
  `h1` read from the nearest date in the sibling
  `modal_analysis/modes_parameters.csv`. Set `isotherm_h1` to override the
  positive depth in metres; `isotherm_stride` controls horizontal sampling
  (default 2). Color shows elevation anomaly relative to the median surface
  height: red indicates sinking and blue indicates rising.

Axes remain in model coordinates; no geographic rotation is applied to either
geometry or velocities. The full lakebed is shown even when `max_depth`
limits eddy layers and arrows. Vertical exaggeration is explicit in the title.

Machine-specific MITgcm settings come from `../config.json`. The default
catalogue path is the sibling `outputs_swirl/eddy_catalogues_final/lvl0.csv`
of the configured outputs directory. Override `catalogue` for other layouts.
An exact model snapshot and catalogue timestamp are required; the script
never silently substitutes a different time. Grid indices and depths are
checked against MITgcm. Eddy cells falling on dry model cells are clipped;
the excluded count is reported in a warning and in the figure title.

The first run scans the entire large catalogue with bounded memory. A small
snapshot cache is reused on subsequent runs and invalidated when the source
file changes. Generated HTML and cache files stay under `results_3d/`.

Dependencies: `numpy`, `pandas`, `matplotlib`, `plotly>=5.21`, `MITgcmutils`.
The notebook can also use the adjacent MITgcm source checkout for MITgcmutils.
For a Python environment with the dependencies installed, the CLI is:

```bash
python plot_3d_eddies.py --time '2025-09-17 00:30:00' --model lucerne_2025
```

## Figure size and optional layers

The notebook defaults to 1,000 pixels tall and fills the available width.
Set `figure_height` and optionally `figure_width` to change this.
Four independent settings control which layers are built:

- `show_velocity`: colored horizontal speed slices at `arrow_depths` (m/s).
- `show_arrows`: horizontal current direction/speed arrows.
- `show_eddies`: catalogue cell sheets; skips catalogue access when false.
- `show_thermocline`: the existing median-temperature isotherm proxy; skips
  temperature processing and modal h1 lookup when false.

Enabled layers can also be hidden and shown through the legend. Both eddy
rotations form one toggle group, as do all velocity slices. These settings
also apply to the date slider. Thermocline and speed colorbars occupy
separate positions. The thermocline proxy retains the existing isotherm
calculation; it is not a new local maximum-gradient thermocline estimate.

Eddy sheets now include side walls along each detected footprint boundary.
Walls span the corresponding MITgcm `RF` cell interfaces and stop at the
lakebed in partial bottom cells. Internal horizontal cell edges have no walls.
Opacity is 0.65 with face shading, making layers easier to see edge-on.
These walls depict the grid-layer thickness, not an interpolated eddy volume.

The upper panel plots total lake kinetic energy in MJ from
`outputs_swirl/ke_eddy/ke_lake.csv`, relative to the configured model directory.
It retains every CSV sample in the selected date window. A red marker follows
the exact displayed snapshot during date-slider changes and playback. Missing
or nonfinite energy at a displayed date raises an error rather than substituting
a nearby value. A zero-day window shows only the selected snapshot. The panel
adds 230 pixels to the requested figure height.
The same panel includes eddy KE (orange) on a separate right-hand MJ axis,
using `ke_eddy.csv` when present or `ke_eddies.csv` otherwise, with column
`kinetic_energy_eddy_[MJ]`. Lake KE remains on the blue left axis and retains
the synchronized red date marker. Both curves use the same selected window.
