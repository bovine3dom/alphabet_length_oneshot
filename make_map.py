#!/usr/bin/env python3
"""Draw a European choropleth from a CSV file and JSON metadata."""

import argparse
import csv
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import FuncNorm, LinearSegmentedColormap, Normalize
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text
from pyproj import Transformer
from shapely import orient_polygons, segmentize, transform
from shapely.geometry import box, shape
from shapely.plotting import patch_from_polygon

from prepare_geography import BOUNDS, OUTPUT as GEOGRAPHY, ROOT, prepare
from prepare_cartogram import BOUNDS as CARTOGRAM_BOUNDS, OUTPUT as CARTOGRAM

SPECTRAL = [
    "#9e0142", "#d53e4f", "#f46d43", "#fdae61", "#fee08b",
    "#ffffbf", "#e6f598", "#abdda4", "#66c2a5", "#3288bd", "#5e4fa2",
]
CMAP = LinearSegmentedColormap.from_list("Spectral", SPECTRAL, N=4096)
GEOMETRIES = ("natural-earth", "owid")
# Small-state point symbols and callouts.
CALLOUTS = {
    "AND": ((1.52, 42.51), (2.6, 43.15)),
    "LIE": ((9.55, 47.17), (9.9, 48.05)),
    "MCO": ((7.42, 43.73), (6.3, 44.15)),
    "MLT": ((14.38, 35.90), (15.4, 36.25)),
    "SMR": ((12.46, 43.94), (13.3, 44.65)),
    "VAT": ((12.45, 41.90), (13.8, 41.5)),
}
BACKGROUND = "#dce3df"
INK = "#000000"
MISSING = {"", "NA", "N/A", "NAN", ".."}
DPI = 120
FONT_SCALE = 1.7
SPLEEN = {height: ROOT / f"fonts/spleen/spleen-{width}x{height}.otf"
          for width, height in ((8, 16), (12, 24), (32, 64))}


def country_aliases(countries):
    aliases = {country["ADM0_A3"]: country["ADM0_A3"] for country in countries}
    for country in countries:
        # Overseas features can share ISO codes with their parent country (e.g. France).
        canonical = aliases.get(country["ISO_A3_EH"], country["ADM0_A3"])
        for field in ("ISO_A2_EH", "ISO_A3_EH"):
            code = country[field]
            if code and code != "-99":
                aliases[code] = canonical
    aliases.update({"UK": "GBR", "EL": "GRC", "XKX": "KOS"})
    return aliases


def detect_country_column(columns, rows, aliases):
    scores = {column: sum((row[column] or "").strip().upper() in aliases for row in rows)
              for column in columns}
    best = max(scores.values(), default=0)
    candidates = [column for column, score in scores.items() if score == best and score > 0]
    if not candidates:
        raise ValueError("CSV requires a column containing country codes")
    if len(candidates) > 1:
        raise ValueError(f"Multiple country-code columns detected: {', '.join(candidates)}")
    return candidates[0]


def load_values(path, metadata, aliases):
    value_column = metadata.get("value", "value")
    values, records = {}, {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {value_column}
        required.update(column["column"] for column in metadata.get("key", {}).get("columns", []))
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV requires columns: {sorted(required)}")
        rows = list(reader)
        code_column = detect_country_column(
            [column for column in reader.fieldnames if column != value_column], rows, aliases,
        )
        for row_number, row in enumerate(rows, 2):
            raw_code = (row[code_column] or "").strip().upper()
            code = aliases.get(raw_code)
            if code is None:
                print(f"Warning: skipping unmatched country code {raw_code!r} "
                      f"at CSV row {row_number} ({code_column}).", file=sys.stderr)
                continue
            if code in values:
                raise ValueError(f"Duplicate country at CSV row {row_number}: {code}")
            records[code] = row
            raw = (row[value_column] or "").strip()
            if raw.upper() in MISSING:
                values[code] = None
                continue
            try:
                value = float(raw)
            except ValueError as error:
                raise ValueError(f"Invalid value at CSV row {row_number}: {raw!r}") from error
            if not np.isfinite(value):
                raise ValueError(f"Value must be finite at CSV row {row_number}")
            values[code] = value
    return values, records


def make_norm(values, scale):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("The map requires at least one finite observation")
    method = scale.get("method", "linear")
    if method not in ("linear", "quantile"):
        raise ValueError("scale.method must be linear or quantile")
    domain = scale.get("domain")
    if domain is not None:
        if method != "linear":
            raise ValueError("scale.domain applies to the linear scale")
        domain = np.asarray(domain, dtype=float)
        if domain.shape != (2,) or not np.isfinite(domain).all() or domain[0] >= domain[1]:
            raise ValueError("scale.domain requires two finite, increasing values")
        return Normalize(*domain, clip=True)
    unique, counts = np.unique(values, return_counts=True)
    if len(unique) == 1:
        return Normalize(unique[0] - 0.5, unique[0] + 0.5, clip=True)
    if method == "linear":
        return Normalize(unique[0], unique[-1], clip=True)
    # Interpolate empirical mid-ranks. Ties share a rank; endpoints use the full palette.
    ranks = np.cumsum(counts) - counts / 2
    ranks = (ranks - ranks[0]) / (ranks[-1] - ranks[0])
    return FuncNorm(
        (lambda x: np.interp(x, unique, ranks), lambda p: np.interp(p, ranks, unique)),
        vmin=unique[0], vmax=unique[-1], clip=True,
    )


def spectral(scale):
    reverse = scale.get("reverse", False)
    if not isinstance(reverse, bool):
        raise ValueError("scale.reverse must be true or false")
    return CMAP.reversed() if reverse else CMAP


def load_map(metadata_path, geometry=None):
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for field in ("data", "title", "source"):
        if not metadata.get(field):
            raise ValueError(f"Metadata requires {field}")
    geometry = geometry or metadata.get("geometry", "natural-earth")
    if geometry not in GEOMETRIES:
        raise ValueError(f"geometry must be one of: {', '.join(GEOMETRIES)}")
    metadata["geometry"] = geometry
    path = CARTOGRAM if geometry == "owid" else GEOGRAPHY
    if geometry == "natural-earth" and not path.exists():
        prepare()
    geography = json.loads(path.read_text(encoding="utf-8"))
    aliases = country_aliases(geography["countries"])
    values, records = load_values(metadata_path.parent / metadata["data"], metadata, aliases)
    visible = {feature["properties"]["ADM0_A3"] for feature in geography["features"]}
    # Use geographic anchors for the same European selection in both layouts.
    visible &= {country["ADM0_A3"] for country in geography["countries"]
                if BOUNDS[0] <= country["LABEL_X"] <= BOUNDS[2]
                and BOUNDS[1] <= country["LABEL_Y"] <= BOUNDS[3]}
    values = {code: value for code, value in values.items() if code in visible and value is not None}
    norm = make_norm(list(values.values()), metadata.get("scale", {}))
    return metadata, geography["features"], values, norm, {code: records[code] for code in values}


def projection():
    return Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True).transform


def graticule(project):
    west, south, east, north = BOUNDS
    for latitude in range(40, north, 10):
        lon = np.linspace(west, east, 400)
        yield project(lon, np.full_like(lon, latitude))
    for longitude in range(-20, east, 10):
        lat = np.linspace(south, north, 400)
        yield project(np.full_like(lat, longitude), lat)


def polygon_patch(geometry, project):
    # Densify crop edges before projection; retain interior rings for lakes and enclaves.
    projected = transform(segmentize(geometry, 0.25), project, interleaved=False)
    return patch_from_polygon(orient_polygons(projected))


def line_segments(geometry):
    if geometry.geom_type in ("LineString", "LinearRing") and not geometry.is_empty:
        yield np.asarray(geometry.coords)
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from line_segments(part)


def legend_ticks(norm):
    lower, upper = np.ceil(norm.vmin), np.floor(norm.vmax)
    if lower > upper:
        return np.array([])
    return np.unique(np.clip(np.rint(norm.inverse(np.linspace(0, 1, 6))), lower, upper))


def format_key_value(value, fmt=None):
    if value is None or str(value).strip().upper() in MISSING:
        return "—"
    if fmt:
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"Key value must be finite: {value!r}")
        return fmt.format(x=number) if "{" in fmt else fmt % number
    return str(value).removesuffix(".0") if isinstance(value, float) else str(value)


def apply_typography(figure, font, dpi, scale):
    for text in figure.findobj(Text):
        text.set_fontsize(text.get_fontsize() * scale)
        if font == "spleen":
            pixels = text.get_fontsize() * dpi / 72
            height = min(SPLEEN, key=lambda size: abs(size - pixels))
            text.set_fontproperties(FontProperties(fname=SPLEEN[height], size=height * 72 / dpi))
            text.set_antialiased(False)


def draw_country_key(fig, features, values, norm, cmap, metadata, records):
    rows = sorted((feature["properties"] for feature in features
                   if feature["properties"]["ADM0_A3"] in values), key=lambda row: row["ADMIN"])
    value_column = metadata.get("value", "value")
    columns = metadata.get("key", {}).get("columns", []) + [
        {"column": value_column, "format": metadata.get("value_format")}
    ]
    ax = fig.add_axes([0.02, 0.10, 0.91, 0.79])
    ax.set_xlim(0, 4.4 + 0.9 * (len(columns) - 1))
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.text(0.33, 1, "COUNTRY", fontsize=9, weight="bold", color=INK)
    for index, column in enumerate(columns):
        ax.text(4.4 + 0.9 * index, 1, column["column"],
                ha="right", fontsize=9, weight="bold", color=INK)
    step = 0.92 / len(rows)
    fontsize = min(9.5, 0.8 * step * ax.bbox.height * 72 / fig.dpi
                   / metadata.get("font_scale", FONT_SCALE))
    for index, row in enumerate(rows):
        y = 0.94 - index * step
        code = row["ADM0_A3"]
        if index % 2 == 0:
            ax.axhspan(y - step / 2, y + step / 2, facecolor="#f4f4f4", edgecolor="none", zorder=-1)
        ax.scatter(0.11, y, s=28, marker="s", c=[cmap(norm(values[code]))],
                   edgecolors=INK, linewidths=0.5)
        cells = [(0.33, row["ISO_A2_EH"], "bold", "left"), (0.792, row["ADMIN"], "normal", "left")]
        for column_index, column in enumerate(columns):
            value = records[code][column["column"]]
            cells.append((4.4 + 0.9 * column_index, format_key_value(value, column.get("format")),
                          "normal", "right"))
        for x, text, weight, align in cells:
            ax.text(x, y, text, va="center", ha=align, fontsize=fontsize, weight=weight, color=INK)


def fit_cartogram_pixels(ax, bounds, dpi):
    """Fit whole display pixels to each cartogram coordinate unit."""
    factor = dpi / ax.figure.dpi
    available = np.array(ax.get_position(original=True).transformed(ax.figure.transSubfigure).bounds) * factor
    span = np.array(bounds[2:]) - bounds[:2]
    scale = np.floor(min(available[2:] / span))
    if scale < 1:
        raise ValueError("Increase dpi to fit at least one pixel per cartogram unit")
    size = span * scale
    origin = np.rint(available[:2] + (available[2:] - size) / 2)
    frame = np.array(ax.figure.bbox.bounds) * factor
    ax.set_position([*((origin - frame[:2]) / frame[2:]), *(size / frame[2:])])


def render(metadata, features, values, norm, output, records):
    cartogram = metadata.get("geometry", "natural-earth") == "owid"
    project = (lambda x, y: (x, y)) if cartogram else projection()
    bounds = CARTOGRAM_BOUNDS if cartogram else BOUNDS
    cmap = spectral(metadata.get("scale", {}))
    font = metadata.get("font", "spleen")
    if font not in ("spleen", "sans"):
        raise ValueError("font must be spleen or sans")
    dpi = metadata.get("dpi", DPI)
    font_scale = metadata.get("font_scale", FONT_SCALE)
    antialias = False
    for name, value in (("dpi", dpi), ("font_scale", font_scale)):
        if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive number")
    # Reserve a fixed-width map panel and add key columns on the right.
    key_width = 4.4 + 0.9 * len(metadata.get("key", {}).get("columns", []))
    figure = plt.figure(figsize=(12.8 + key_width, 10.2), facecolor="#ffffff")
    fig, key = figure.subfigures(1, 2, width_ratios=[12.8, key_width], wspace=0, facecolor="#ffffff")
    fig.text(0.05, 0.95, metadata["title"], fontsize=22, weight="bold", color=INK)
    fig.text(0.05, 0.915, metadata.get("subtitle", ""), fontsize=10, color=INK)
    ax = fig.add_axes([0.10, 0.10, 0.89, 0.79], aspect="equal")
    ax.set_axis_off()
    view = transform(segmentize(box(*bounds), 0.25), project, interleaved=False)
    ax.add_patch(patch_from_polygon(view, facecolor="#f7f6f1", edgecolor="none",
                                    antialiased=antialias, zorder=-1))
    west, south, east, north = view.bounds
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    if cartogram:
        fit_cartogram_pixels(ax, bounds, dpi)
    if not cartogram:
        for x, y in graticule(project):
            ax.plot(x, y, color="#d5e0e1", linewidth=0.5, antialiased=antialias, zorder=0)

    patches, colors, borders = [], [], []
    for feature in features:
        code = feature["properties"]["ADM0_A3"]
        patches.append(polygon_patch(shape(feature["geometry"]), project))
        colors.append(cmap(norm(values[code])) if code in values else BACKGROUND)
        boundary = transform(segmentize(shape(feature["boundary"]), 0.25), project, interleaved=False)
        borders.extend(line_segments(boundary))
    ax.add_collection(PatchCollection(patches, facecolors=colors, edgecolors="none",
                                      linewidths=0, antialiased=antialias,
                                      snap=True if cartogram else None, zorder=2))
    ax.add_collection(LineCollection(borders, colors="#777777",
                                     linewidths=2 * 72 / dpi if cartogram else 0.28,
                                     antialiaseds=antialias, snap=True if cartogram else None,
                                     capstyle="butt", zorder=3))

    for feature in features:
        properties = feature["properties"]
        code = properties["ADM0_A3"]
        if code not in values:
            continue
        lon, lat = properties["LABEL_X"], properties["LABEL_Y"]
        if not cartogram and code in CALLOUTS:
            point, (lon, lat) = CALLOUTS[code]
            x, y = project(*point)
            tx, ty = project(lon, lat)
            ax.scatter(x, y, s=28, c=[cmap(norm(values[code]))], edgecolors="white",
                       linewidths=0.8, antialiaseds=antialias, zorder=6)
            ax.plot([x, tx], [y, ty], color="#52646a", linewidth=0.55,
                    antialiased=antialias, zorder=5)
        if not (bounds[0] <= lon <= bounds[2] and bounds[1] <= lat <= bounds[3]):
            continue
        label = ax.text(*project(lon, lat), properties["ISO_A2_EH"], ha="center", va="center",
                        fontsize=6, weight="bold", color=INK, antialiased=antialias, zorder=7)
        label.set_path_effects([path_effects.withStroke(linewidth=1, foreground="white",
                                                       antialiased=antialias)])

    cax = fig.add_axes([0.035, 0.18, 0.018, 0.64])
    observations = np.asarray(list(values.values()))
    below, above = observations.min() < norm.vmin, observations.max() > norm.vmax
    extend = "both" if below and above else "min" if below else "max" if above else "neither"
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                            orientation="vertical", ticks=legend_ticks(norm), extend=extend,
                            format="%.0f")
    colorbar.outline.set(visible=True, edgecolor=INK, linewidth=0.5)
    colorbar.ax.tick_params(labelsize=8, length=2, colors=INK, pad=4)
    fig.text(0.05, 0.058, textwrap.fill(metadata.get("note", ""), 145), fontsize=7, color=INK)
    attribution = ("Layout: Our World in Data uniform population cartogram (2026)." if cartogram
                   else "Boundaries: Natural Earth 1:10m")
    source = f"{metadata['source']}. {attribution}"
    figure.text(0.05 * 12.8 / (12.8 + key_width), 0.025, source,
                fontsize=6.5, color=INK, wrap=True)
    draw_country_key(key, features, values, norm, cmap, metadata, records)
    output.parent.mkdir(parents=True, exist_ok=True)
    apply_typography(figure, font, dpi, font_scale)
    with matplotlib.rc_context({"text.hinting": "none"} if font == "spleen" else {}):
        figure.savefig(output, dpi=dpi, facecolor=figure.get_facecolor())
    plt.close(figure)
    print(f"Wrote {output} ({len(values)} countries; {metadata['geometry']})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path, nargs="?",
                        default=ROOT / "data/expected_career_duration_oecd_mean.json",
                        help="JSON metadata (default: bundled example)")
    parser.add_argument("--output", type=Path, help="Output PNG path")
    parser.add_argument("--geometry", choices=GEOMETRIES, help="Override metadata geometry")
    args = parser.parse_args()
    try:
        metadata, features, values, norm, records = load_map(args.metadata, args.geometry)
        suffix = "_owid" if metadata["geometry"] == "owid" else ""
        output = args.output or ROOT / "output" / f"{args.metadata.stem}{suffix}.png"
        render(metadata, features, values, norm, output, records)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
