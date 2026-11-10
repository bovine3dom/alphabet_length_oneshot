#!/usr/bin/env python3
"""Build a European alphabet-length map from Natural Earth 1:110m polygons."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SOURCE = DATA / "ne_110m_admin_0_countries.geojson"
CROPPED = DATA / "ne_110m_europe_crop.geojson"
OUTPUT = ROOT / "europe_alphabet_map.png"
CSV = ROOT / "alphabet_counts.csv"

# D3-scale-chromatic schemeSpectral[11], interpolated continuously.
SPECTRAL = [
    "#9e0142", "#d53e4f", "#f46d43", "#fdae61", "#fee08b",
    "#ffffbf", "#e6f598", "#abdda4", "#66c2a5", "#3288bd", "#5e4fa2",
]
CMAP = LinearSegmentedColormap.from_list("D3 Spectral", SPECTRAL, N=256)

# name, Natural Earth ADMIN name (None for point-scale countries), ISO-style code,
# representative language, alphabet entries, map label position (longitude, latitude).
COUNTRIES = [
    ("Albania", "Albania", "AL", "Albanian", 36, (19.6, 40.7)),
    ("Andorra", None, "AD", "Catalan", 35, (2.7, 43.1)),
    ("Armenia", "Armenia", "AM", "Armenian", 39, (44.4, 39.0)),
    ("Austria", "Austria", "AT", "German", 30, (14.4, 48.2)),
    ("Azerbaijan", "Azerbaijan", "AZ", "Azerbaijani", 32, (46.5, 41.3)),
    ("Belarus", "Belarus", "BY", "Belarusian", 32, (28.1, 53.4)),
    ("Belgium", "Belgium", "BE", "Dutch", 26, (3.5, 50.5)),
    ("Bosnia and Herzegovina", "Bosnia and Herzegovina", "BA", "Bosnian", 30, (18.4, 44.0)),
    ("Bulgaria", "Bulgaria", "BG", "Bulgarian", 30, (25.2, 42.8)),
    ("Croatia", "Croatia", "HR", "Croatian", 30, (15.8, 45.3)),
    ("Cyprus", "Cyprus", "CY", "Greek", 33, (33.3, 34.8)),
    ("Czechia", "Czechia", "CZ", "Czech", 42, (15.6, 50.0)),
    ("Denmark", "Denmark", "DK", "Danish", 29, (10.1, 56.0)),
    ("Estonia", "Estonia", "EE", "Estonian", 27, (25.7, 58.9)),
    ("Finland", "Finland", "FI", "Finnish", 29, (27.4, 63.5)),
    ("France", "France", "FR", "French", 39, (1.8, 46.3)),
    ("Georgia", "Georgia", "GE", "Georgian", 33, (41.7, 42.7)),
    ("Germany", "Germany", "DE", "German", 30, (10.2, 51.3)),
    ("Greece", "Greece", "GR", "Greek", 33, (23.2, 38.8)),
    ("Hungary", "Hungary", "HU", "Hungarian", 44, (19.3, 47.0)),
    ("Iceland", "Iceland", "IS", "Icelandic", 32, (-18.5, 65.1)),
    ("Ireland", "Ireland", "IE", "English", 26, (-8.0, 53.3)),
    ("Italy", "Italy", "IT", "Italian", 28, (12.2, 42.7)),
    ("Kosovo", "Kosovo", "XK", "Albanian", 36, (22.0, 43.1)),
    ("Latvia", "Latvia", "LV", "Latvian", 33, (24.7, 57.1)),
    ("Liechtenstein", None, "LI", "German", 30, (9.8, 48.0)),
    ("Lithuania", "Lithuania", "LT", "Lithuanian", 32, (24.0, 55.2)),
    ("Luxembourg", "Luxembourg", "LU", "Luxembourgish", 29, (6.2, 49.5)),
    ("Malta", None, "MT", "Maltese", 30, (15.6, 36.2)),
    ("Moldova", "Moldova", "MD", "Romanian", 31, (29.0, 47.5)),
    ("Monaco", None, "MC", "French", 39, (6.3, 44.1)),
    ("Montenegro", "Montenegro", "ME", "Montenegrin", 32, (18.5, 42.7)),
    ("Netherlands", "Netherlands", "NL", "Dutch", 26, (5.4, 52.2)),
    ("North Macedonia", "North Macedonia", "MK", "Macedonian", 31, (22.2, 41.2)),
    ("Norway", "Norway", "NO", "Norwegian", 29, (13.3, 63.4)),
    ("Poland", "Poland", "PL", "Polish", 32, (19.2, 52.3)),
    ("Portugal", "Portugal", "PT", "Portuguese", 38, (-8.2, 39.4)),
    ("Romania", "Romania", "RO", "Romanian", 31, (25.0, 45.7)),
    ("Russia", "Russia", "RU", "Russian", 33, (39.5, 59.5)),
    ("San Marino", None, "SM", "Italian", 28, (13.3, 44.6)),
    ("Serbia", "Republic of Serbia", "RS", "Serbian", 30, (21.5, 44.5)),
    ("Slovakia", "Slovakia", "SK", "Slovak", 46, (19.1, 48.7)),
    ("Slovenia", "Slovenia", "SI", "Slovene", 25, (14.0, 45.8)),
    ("Spain", "Spain", "ES", "Spanish", 33, (-3.5, 40.4)),
    ("Sweden", "Sweden", "SE", "Swedish", 29, (17.5, 61.4)),
    ("Switzerland", "Switzerland", "CH", "German", 30, (7.7, 46.6)),
    ("Turkey", "Turkey", "TR", "Turkish", 29, (35.0, 39.1)),
    ("Ukraine", "Ukraine", "UA", "Ukrainian", 33, (31.4, 49.0)),
    ("United Kingdom", "United Kingdom", "GB", "English", 26, (-2.7, 54.7)),
    ("Vatican City", None, "VA", "Italian", 28, (13.8, 41.5)),
]

# Point locations for small states omitted at Natural Earth's 1:110m scale.
POINTS = {
    "AD": (1.52, 42.51), "LI": (9.55, 47.17), "MC": (7.42, 43.73),
    "MT": (14.38, 35.90), "SM": (12.46, 43.94), "VA": (12.45, 41.90),
}

# Spherical Lambert conformal conic projection, centered on Europe.
def project(lon, lat):
    phi = np.radians(lat)
    lam = np.radians(lon)
    phi1, phi2 = np.radians(43.0), np.radians(62.0)
    phi0, lam0 = np.radians(52.0), np.radians(15.0)
    n = np.log(np.cos(phi1) / np.cos(phi2)) / np.log(
        np.tan(np.pi / 4 + phi2 / 2) / np.tan(np.pi / 4 + phi1 / 2)
    )
    f = np.cos(phi1) * np.tan(np.pi / 4 + phi1 / 2) ** n / n
    rho = f / np.tan(np.pi / 4 + phi / 2) ** n
    rho0 = f / np.tan(np.pi / 4 + phi0 / 2) ** n
    theta = n * (lam - lam0)
    return rho * np.sin(theta), rho0 - rho * np.cos(theta)


def polygons(feature):
    geometry = feature.get("geometry")
    if not geometry:
        return
    coords = geometry["coordinates"]
    if geometry["type"] == "Polygon":
        yield coords
    elif geometry["type"] == "MultiPolygon":
        yield from coords


def projected_rings(feature):
    for polygon in polygons(feature):
        ring = polygon[0]
        lon, lat = np.asarray(ring).T
        x, y = project(lon, lat)
        yield np.column_stack((x, y))


def main():
    if len(COUNTRIES) != 50:
        raise ValueError(f"Expected 50 countries, found {len(COUNTRIES)}")
    features = json.loads(CROPPED.read_text(encoding="utf-8"))["features"]
    by_admin = {f["properties"]["ADMIN"]: f for f in features}
    missing = [admin for _, admin, _, _, _, _ in COUNTRIES if admin and admin not in by_admin]
    if missing:
        raise ValueError(f"Missing Natural Earth countries: {missing}")

    counts = [row[4] for row in COUNTRIES]
    norm = Normalize(min(counts), max(counts))
    with CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["country", "code", "representative_language", "alphabet_entries"])
        writer.writerows((name, code, language, count) for name, _, code, language, count, _ in COUNTRIES)

    fig = plt.figure(figsize=(16.4, 10.0), facecolor="#f7f6f1")
    fig.text(0.035, 0.955, "EUROPE, LETTER BY LETTER", fontsize=24, weight="bold",
             color="#152b3c", family="DejaVu Sans")
    fig.text(0.037, 0.917,
             "Alphabet length in one principal language per country  ·  50 countries  ·  1:110m geography",
             fontsize=9.5, color="#4d626b")
    fig.text(0.965, 0.95, "LANGUAGE ATLAS  /  01", ha="right", fontsize=7,
             weight="bold", color="#6d7d80")

    ax = fig.add_axes([0.025, 0.12, 0.665, 0.755])
    ax.set_facecolor("#eaf1f2")
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

    # Subtle graticule for geographic context.
    for latitude in (40, 50, 60, 70):
        lon = np.linspace(-28, 48, 240)
        x, y = project(lon, np.full_like(lon, latitude))
        ax.plot(x, y, color="#d5e0e1", linewidth=0.5, zorder=0)
    for longitude in (-20, -10, 0, 10, 20, 30, 40):
        lat = np.linspace(32, 74, 180)
        x, y = project(np.full_like(lat, longitude), lat)
        ax.plot(x, y, color="#d5e0e1", linewidth=0.5, zorder=0)

    background_rings = [
        ring for feature in features
        if feature["properties"]["ADMIN"] != "Greenland"
        for ring in projected_rings(feature)
    ]
    ax.add_collection(PolyCollection(
        background_rings, closed=True, facecolors="#dce3df", edgecolors="#a8b7b8",
        linewidths=0.36, zorder=1, antialiased=True,
    ))

    for name, admin, code, language, count, _ in COUNTRIES:
        if admin is None:
            continue
        rings = list(projected_rings(by_admin[admin]))
        ax.add_collection(PolyCollection(
            rings, closed=True, facecolors=[CMAP(norm(count))], edgecolors="#ffffff",
            linewidths=0.7, zorder=2, antialiased=True,
        ))

    # Fit a projected grid over the clipped European view.
    grid_lon, grid_lat = np.meshgrid(np.linspace(-28, 48, 120), np.linspace(32, 74, 100))
    gx, gy = project(grid_lon, grid_lat)
    ax.set_xlim(gx.min(), gx.max())
    ax.set_ylim(gy.min(), gy.max())

    # Short country identifiers link the map to the language/count key.
    label_offsets = {
        "AL": (-0.2, -0.25), "BA": (0.45, -0.1), "BE": (-0.2, -0.35),
        "CZ": (0.15, 0.25), "HU": (-0.2, -0.1), "LU": (0.45, -0.1),
        "ME": (-0.25, 0.25), "MK": (0.7, -0.25), "NL": (0.25, 0.15),
        "RS": (0.45, 0.2), "SI": (-0.4, -0.2), "SK": (0.35, 0.2),
        "XK": (0.75, 0.2), "AZ": (0.25, 0.25), "GE": (-0.3, 0.4),
        "AM": (0.0, -0.3),
    }
    for name, admin, code, language, count, anchor in COUNTRIES:
        if code in POINTS:
            continue
        lon, lat = anchor
        dx, dy = label_offsets.get(code, (0.0, 0.0))
        x, y = project(lon + dx, lat + dy)
        label = ax.text(x, y, code, ha="center", va="center", fontsize=6.2,
                        weight="bold", color="#172d3c", zorder=5)
        label.set_path_effects([path_effects.withStroke(linewidth=2.0, foreground="#ffffff")])

    # Point symbols identify states that Natural Earth's low-resolution polygons omit.
    point_labels = {
        "AD": (2.6, 43.15), "LI": (9.9, 48.05), "MC": (6.3, 44.15),
        "MT": (15.4, 36.25), "SM": (13.3, 44.65), "VA": (13.8, 41.5),
    }
    for name, admin, code, language, count, _ in COUNTRIES:
        if code not in POINTS:
            continue
        lon, lat = POINTS[code]
        x, y = project(lon, lat)
        ax.scatter([x], [y], s=30, marker="o", c=[CMAP(norm(count))],
                   edgecolors="#ffffff", linewidths=0.8, zorder=6)
        label_lon, label_lat = point_labels[code]
        tx, ty = project(label_lon, label_lat)
        ax.plot([x, tx], [y, ty], color="#52646a", linewidth=0.55, zorder=5)
        label = ax.text(tx, ty, code, ha="center", va="center", fontsize=6.2,
                        weight="bold", color="#172d3c", zorder=7)
        label.set_path_effects([path_effects.withStroke(linewidth=2.0, foreground="#ffffff")])

    # Country/language key with a color swatch for every mapped value.
    panel = Rectangle((0.713, 0.105), 0.275, 0.785, transform=fig.transFigure,
                      facecolor="#f0f0e9", edgecolor="#d9ded8", linewidth=0.8, zorder=0)
    fig.add_artist(panel)
    fig.text(0.728, 0.858, "COUNTRY  /  LANGUAGE", fontsize=8, weight="bold", color="#304953")
    fig.text(0.728, 0.836, "Code · representative language · alphabet entries", fontsize=6.8, color="#718087")
    left_x, right_x = 0.729, 0.86
    columns = (COUNTRIES[:25], COUNTRIES[25:])
    for col, column in enumerate(columns):
        x0 = (left_x, right_x)[col]
        for i, (name, _, code, language, count, _) in enumerate(column):
            y = 0.805 - i * 0.0277
            fig.add_artist(Rectangle((x0, y - 0.004), 0.0065, 0.014,
                                     transform=fig.transFigure, facecolor=CMAP(norm(count)),
                                     edgecolor="none", zorder=2))
            short_name = name.replace("Bosnia and Herzegovina", "Bosnia & Herzegovina")
            label = f"{code}  {short_name} · {language} · {count}"
            fig.text(x0 + 0.009, y, label, fontsize=5.45, color="#203640", va="center")

    # Continuous legend uses D3's Spectral palette and the observed count range.
    cax = fig.add_axes([0.075, 0.063, 0.54, 0.016])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax,
                      orientation="horizontal", ticks=[25, 30, 35, 40, 46])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5, length=2, colors="#53656d", pad=2)
    fig.text(0.075, 0.086, "ALPHABET ENTRIES", fontsize=6.5, weight="bold", color="#53656d")
    fig.text(0.615, 0.071, "D3 SPECTRAL", fontsize=6.5, weight="bold", color="#53656d", ha="right")
    fig.text(0.035, 0.025,
             "Source: Natural Earth, 1:110m Admin 0 – Countries (public domain; naturalearthdata.com).  "
             "Counts are illustrative; common accented forms count separately and established digraphs count once. "
             "French follows the W-excluded convention. Point markers show states omitted at this map scale.",
             fontsize=5.35, color="#64747a")

    fig.savefig(OUTPUT, dpi=200, facecolor=fig.get_facecolor(), bbox_inches=None)
    plt.close(fig)
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")
    print(f"Wrote {CSV}")


if __name__ == "__main__":
    main()
