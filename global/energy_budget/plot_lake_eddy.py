"""Create a standalone Bokeh chart of lake and eddy KE using config.json paths."""
from pathlib import Path

import pandas as pd
from bokeh.embed import file_html
from bokeh.models import HoverTool
from bokeh.resources import INLINE

from analysis import data_paths, select_period
from plots import interactive


def main():
    model = "zug_2025"
    paths = data_paths(model)
    data = {}
    for key, column in (("total_ke", "kinetic_energy_[MJ]"),
                        ("eddy_ke", "kinetic_energy_eddy_[MJ]")):
        frame = pd.read_csv(paths[key], usecols=["date", column], parse_dates=["date"])
        data[key] = frame.set_index("date")[column].sort_index()
    data = select_period(data, "2025-03-01", "2026-03-01")
    if any(series.empty for series in data.values()):
        raise ValueError("No KE data in the selected date interval.")
    plot = interactive(data, ["total_ke", "eddy_ke"])
    plot.title.text = "Lake Zug — lake and eddy kinetic energy (0–60 m)"
    plot.sizing_mode = "stretch_width"
    for renderer, label in zip(plot.renderers, ["Lake KE", "Eddy KE"]):
        renderer.glyph.line_width = 2
        plot.add_tools(HoverTool(
            renderers=[renderer], mode="vline",
            tooltips=[("Series", label), ("Date", "@x{%F %H:%M}"), ("KE [MJ]", "@y{0,0.00}")],
            formatters={"@x": "datetime"},
        ))
    output = Path(__file__).resolve().parent / "figures" / "lake_eddy_ke.html"
    output.parent.mkdir(exist_ok=True)
    output.write_text(file_html(plot, INLINE, "Lake Zug — lake and eddy KE"), encoding="utf-8")
    print(output)
    for key, series in data.items():
        print(f"{key}: {len(series)} points, {series.index.min()} to {series.index.max()}")


if __name__ == "__main__":
    main()
