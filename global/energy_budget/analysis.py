"""Energy-budget CSV loading and analyses; importing this module does not read data."""
from pathlib import Path
import json
import socket

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def data_paths(model="zug_2025", *, hostname=None, depth_suffix="_-0.25--56.6m",
               seiche_folder=None):
    """Resolve the original notebook's CSV locations using the project config."""
    hostname = hostname or socket.gethostname()
    with (PROJECT_ROOT / "config.json").open() as stream:
        config = json.load(stream)
    if hostname not in config or model not in config[hostname]:
        raise KeyError(f"No configuration for {hostname!r}/{model!r} in config.json")
    root = Path(config[hostname][model]["datapath"]).parent
    budget = root / "energy_budget"
    lake = model.rsplit("_", 1)[0]
    seiche = (Path(seiche_folder) if seiche_folder is not None else
              PROJECT_ROOT.parent / "modal_analysis" / "horizontal_mode_analysis" /
              "figures" / "modal_decomposition" / lake)
    return {
        "total_ke": budget / f"ke_lake{depth_suffix}.csv",
        "eddy_ke": budget / f"ke_eddies{depth_suffix}.csv",
        "total_pe": budget / "EP_KBWinters1995.csv",
        "seiche_ke": seiche / "KE_V1H1.csv",
        "wind_energy": budget / "E_wind_MJperh.csv",
        "p10": root / "wind_analysis" / "P10.csv",
    }


def load_data(paths, *, include_wind=False, include_p10=False):
    """Read selected CSVs once. Return independently indexed series in MJ (rates in MJ/h)."""
    columns = {
        "total_ke": ("date", "kinetic_energy_[MJ]"),
        "eddy_ke": ("date", "kinetic_energy_eddy_[MJ]"),
        "total_pe": ("time", "APE"),
        "seiche_ke": ("time", "KE_mode_MJ"),
    }
    if include_wind:
        columns["wind_energy"] = ("time", "E_wind_MJperh")
    result = {}
    for name, (date, value) in columns.items():
        frame = pd.read_csv(paths[name], usecols=[date, value])
        series = pd.Series(pd.to_numeric(frame[value]).to_numpy(),
                           index=pd.to_datetime(frame[date]), name=name).sort_index()
        if series.index.has_duplicates:
            raise ValueError(f"Duplicate timestamps in {paths[name]}")
        result[name] = series / 1e6 if name == "total_pe" else series
    if include_p10:
        frame = pd.read_csv(paths["p10"], index_col=0)
        series = pd.to_numeric(frame["P10_MJperh"])
        series.index = pd.to_datetime(series.index)
        if series.index.has_duplicates:
            raise ValueError(f"Duplicate timestamps in {paths['p10']}")
        result["p10"] = series.sort_index().rename("p10")
    return result


def select_period(data, start, end):
    """Select an open interval, as in the source notebook's fraction calculations."""
    return {name: s.loc[(s.index > pd.Timestamp(start)) & (s.index < pd.Timestamp(end))]
            for name, s in data.items()}


def energy_fractions(data, frequency="1h"):
    """Resample to shared bins; missing observations stay missing, zero KE gives NaN."""
    frame = pd.concat({k: data[k].resample(frequency).mean()
                       for k in ("total_ke", "eddy_ke", "seiche_ke")}, axis=1)
    denominator = frame["total_ke"].replace(0, np.nan)
    ratios = frame[["eddy_ke", "seiche_ke"]].div(denominator, axis=0) * 100
    means = {}
    for key in ratios:
        paired = frame[["total_ke", key]].dropna()
        total = paired["total_ke"].sum()
        means[key] = 100 * paired[key].sum() / total if total else np.nan
    return ratios, pd.Series(means, name="Fraction of summed KE [%]")


def monthly_energy(data):
    frame = pd.concat({k: data[k].resample("MS").mean()
                       for k in ("total_ke", "eddy_ke", "seiche_ke")}, axis=1)
    frame["residual"] = frame.total_ke - frame.eddy_ke - frame.seiche_ke
    return frame


def hourly_wind(data):
    """Mean power per hourly bin in MJ/h; missing bins remain NaN."""
    return data["wind_energy"].resample("1h").mean()


def wind_input_past_day(data):
    """Integrate 24 complete hourly bins of wind power to MJ."""
    return hourly_wind(data).rolling(24, min_periods=24).sum()


def calm_periods(data, threshold=1e3):
    """Hourly calm runs and eddy-fraction change from the preceding windy hour.

    Missing wind or fraction observations break a run. Changes stay unknown until
    a windy baseline is observed. Threshold is in MJ/h; changes are percentage points.
    """
    ratios, _ = energy_fractions(data)
    frame = pd.concat([hourly_wind(data).rename("wind"),
                       ratios.eddy_ke.rename("eddy_fraction")], axis=1)
    hours, changes = [], []
    count, baseline = 0, np.nan
    for wind, fraction in frame.itertuples(index=False, name=None):
        if pd.isna(wind) or pd.isna(fraction):
            count, baseline = 0, np.nan
            hours.append(np.nan)
            changes.append(np.nan)
            continue
        if wind >= threshold:
            count, baseline = 0, fraction
        else:
            count += 1
        hours.append(count)
        changes.append(fraction - baseline)
    frame["hours_without_wind"] = hours
    frame["eddy_fraction_change"] = changes
    return frame


def fraction_correlations(ratios, hours=(12, 24, 36, 48, 72), lag_frequency="72h"):
    correlations = {h: ratios.resample(f"{h}h").mean().eddy_ke.corr(
        ratios.resample(f"{h}h").mean().seiche_ke) for h in hours}
    sampled = ratios.resample(lag_frequency).mean()
    lagged = {lag: sampled.seiche_ke.corr(sampled.eddy_ke.shift(lag))
              for lag in range(-10, 11)}
    return (pd.Series(correlations, name="Pearson correlation").rename_axis("Bin [hours]"),
            pd.Series(lagged, name="Pearson correlation").rename_axis(
                f"Eddy shift [bins of {lag_frequency}]"))


def cross_correlation(data):
    """Original normalized full correlation, on complete hourly KE series.

    Positive lag means eddy KE follows total KE. Reject gaps rather than compressing time.
    """
    from scipy.signal import correlate, correlation_lags
    frame = pd.concat({k: data[k].resample("1h").mean()
                       for k in ("total_ke", "eddy_ke")}, axis=1)
    if len(frame) < 2 or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Cross-correlation requires at least two complete hourly KE bins; select a gap-free period.")
    x, y = frame.total_ke.to_numpy(), frame.eddy_ke.to_numpy()
    x, y = x - x.mean(), y - y.mean()
    scale = x.std() * y.std() * len(x)
    if scale == 0:
        raise ValueError("Cross-correlation requires nonconstant KE series.")
    return pd.Series(correlate(y, x, mode="full") / scale,
                     index=correlation_lags(len(y), len(x)), name="Correlation").rename_axis("Lag [hours]")


def event_summary(data, start, end):
    """Compare mean energy contents with integrated wind/P10 over an open interval.

    Requires complete hourly rate data inside the interval. Each rate bin represents
    one hour, so summing MJ/h bins gives MJ. These ratios are content/input comparisons,
    not conversion efficiencies.
    """
    selected = select_period(data, start, end)
    expected = pd.date_range(pd.Timestamp(start).floor("h"), pd.Timestamp(end).ceil("h"), freq="1h")
    expected = expected[(expected > pd.Timestamp(start)) & (expected < pd.Timestamp(end))]
    inputs = {}
    for key in ("wind_energy", "p10"):
        hourly = data[key].resample("1h").mean().reindex(expected)
        if hourly.empty or not np.isfinite(hourly.to_numpy()).all():
            raise ValueError(f"{key} needs complete hourly data over the event interval.")
        inputs[key] = hourly.sum()
    p10 = inputs["p10"]
    values = {key: selected[key].mean() for key in ("total_ke", "total_pe", "seiche_ke")}
    values["wind_energy"] = inputs["wind_energy"]
    frame = pd.DataFrame({"Energy [MJ]": values})
    frame["Percent of integrated P10"] = frame["Energy [MJ]"] / (p10 or np.nan) * 100
    return frame, p10
