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
  colored cell sheets at each detection depth (orange clockwise, green
  anticlockwise). These are sampled layers, not interpolated eddy volumes.
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
