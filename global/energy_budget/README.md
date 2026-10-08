# Energy budget

A small working notebook backed by reusable Python functions, based on
[`../energy_budget.ipynb`](../energy_budget.ipynb). The original notebook is preserved.

- `energy_budget.ipynb`: settings, one loading cell, analysis calls, and plots.
- `analysis.py`: config-based paths, CSV loading, fractions, monthly means, wind periods, correlations, and event summaries.
- `plots.py`: Matplotlib figures and optional Bokeh plots.

Open the new notebook with an environment containing pandas, NumPy, Matplotlib,
SciPy, and IPython/Jupyter. Bokeh is optional. Set the model, date interval, depth
suffix, and optional inputs, inspect `paths`, then run the loading cell once.
Subsequent cells reuse `data` in memory. Autoreload picks up edits to the modules;
changes to loading logic require rerunning the loading cell.

## Inputs

See [DATA.md](../../DATA.md) and [config.json](../../config.json). Paths are
resolved from the configuration entry for the selected hostname and model.
Energy CSVs live in `energy_budget/` alongside the configured `outputs/` folder;
P10 lives in the sibling `wind_analysis/` folder. The default seiche folder is
`python/modal_analysis/horizontal_mode_analysis/figures/modal_decomposition/<lake>`
relative to the workspace, matching the original notebook. Override
`seiche_folder` in `data_paths()` or individual entries of `paths` as needed.

The loader expects the same filenames and column names as the source notebook.
APE is converted from J to MJ. Wind and P10 inputs are rates in MJ/h. The default
depth suffixes and labels reproduce Zug and Lucerne settings in that notebook;
set them explicitly for other datasets. No raw MITgcm output is loaded here.

## Calculation choices

- Every analysis uses the selected open date interval (`start < time < end`).
- Fractions use common resampling bins. Mean fractions are ratios of summed
  energies over pairwise available bins, rather than sums at potentially different
  sampling frequencies. Missing seiche values remain missing instead of becoming zero.
- Monthly labels come from the actual dates; the radar polygon closes once and
  does not add an offset to the values. The residual is total minus eddy minus seiche KE.
- Wind calculations use hourly mean rates. The preceding-day input requires 24
  complete bins. Calm periods align wind and fractions by timestamp; missing bins
  break a run. Fraction changes require a preceding windy baseline.
- KE cross-correlation requires complete hourly data and rejects gaps or constant
  series. Positive lag means eddy KE follows total KE. Fraction lag correlations
  explicitly label their bin size (72 hours by default).
- Event wind/P10 sums assume each hourly rate bin represents one hour and require
  complete bins in the open interval. Stored-energy/input ratios are descriptive
  comparisons, not conversion efficiencies.

Wind/P10 analyses and Bokeh plots are optional, avoiding the undefined wind variable
in the original notebook. The unused eddy-volume CSV read and hard-coded literature
arithmetic are omitted. Figures are returned for inspection; saving is explicit.
