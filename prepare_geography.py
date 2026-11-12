#!/usr/bin/env python3
"""Prepare the Natural Earth country data for the European map."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from shapely import make_valid
from shapely.geometry import box, mapping, shape

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data/ne_10m_europe_crop.geojson"
BOUNDS = (-28, 32, 48, 74)
SOURCE = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "v5.1.2/geojson/ne_10m_admin_0_countries.geojson"
)
SHA256 = "239eec57ac17f100a11e2536cffc56752c318b50ae765b0918ff7aab4ce8f255"
FIELDS = ("ADMIN", "ISO_A2_EH", "ISO_A3_EH", "ADM0_A3", "LABEL_X", "LABEL_Y")


def clip_country(geometry, view):
    """Clip the fill and the original boundary lines separately."""
    geometry = make_valid(geometry)
    return geometry.intersection(view), geometry.boundary.intersection(view)


def prepare(output=OUTPUT):
    with urlopen(SOURCE, timeout=60) as response:
        raw = response.read()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError("Natural Earth source checksum mismatch")
    view = box(*BOUNDS)
    countries, features = [], []
    for feature in json.loads(raw)["features"]:
        properties = {key: feature["properties"][key] for key in FIELDS}
        countries.append(properties)
        geometry = shape(feature["geometry"])
        if properties["ADMIN"] == "Greenland" or not geometry.intersects(view):
            continue
        geometry, boundary = clip_country(geometry, view)
        if geometry.geom_type not in ("Polygon", "MultiPolygon") or geometry.is_empty:
            continue
        features.append({
            "type": "Feature", "properties": properties, "geometry": mapping(geometry),
            "boundary": mapping(boundary),
        })
    collection = {
        "type": "FeatureCollection",
        "source": {"url": SOURCE, "sha256": SHA256, "bounds": BOUNDS},
        "countries": countries,
        "features": features,
    }
    output.write_text(json.dumps(collection, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Wrote {output} ({len(features)} country geometries)")


if __name__ == "__main__":
    prepare()
