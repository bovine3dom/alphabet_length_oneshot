#!/usr/bin/env python3
"""Import OWID cells (x,y,numeric code) and an alpha3,numeric country crosswalk."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from shapely import maximum_inscribed_circle, union_all
from shapely.geometry import box, mapping

from prepare_geography import OUTPUT as NATURAL_EARTH, ROOT, clip_country
from prepare_geography import prepare as prepare_natural_earth

OUTPUT = ROOT / "data/owid_europe_cartogram.geojson"
BOUNDS = (340, -330, 620, -100)


def prepare(cells_path, codes_path, output=OUTPUT):
    if not NATURAL_EARTH.exists():
        prepare_natural_earth()
    geography = json.loads(NATURAL_EARTH.read_text(encoding="utf-8"))
    countries = geography["countries"]
    by_code = {country["ISO_A3_EH"]: country for country in countries}
    by_code.update({country["ADM0_A3"]: country for country in countries})
    with codes_path.open(encoding="utf-8") as file:
        codes = {int(row["numeric"]): row["alpha3"] for row in csv.DictReader(file)}
    with cells_path.open(encoding="utf-8", newline="") as file:
        rows = [(int(x), -int(y), int(code)) for x, y, code in csv.reader(file)]
    west, south, east, north = BOUNDS
    visible = {code for x, y, code in rows
               if west - 1 <= x <= east + 1 and south - 1 <= y <= north + 1}
    cells = defaultdict(list)
    for x, y, code in rows:
        if code in visible:
            cells[code].append(box(x - 1, y - 1, x + 1, y + 1))
    view = box(*BOUNDS)
    features = []
    for code, tiles in sorted(cells.items()):
        properties = by_code[codes[code]]
        if properties["ADMIN"] == "Greenland":
            continue
        geometry, boundary = clip_country(union_all(tiles), view)
        if geometry.is_empty or geometry.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        x, y = maximum_inscribed_circle(geometry, tolerance=0.25).coords[0]
        features.append({
            "type": "Feature",
            "properties": properties | {"OWID_CODE": code, "LABEL_X": x, "LABEL_Y": y},
            "geometry": mapping(geometry), "boundary": mapping(boundary),
        })
    collection = {
        "type": "FeatureCollection",
        "source": {
            "url": "https://owid.github.io/cartograms/", "year": 2026,
            "layout": "fluid", "shape": "square", "radius": 1, "bounds": BOUNDS,
            "inputs": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                       for name, path in (("cells", cells_path), ("codes", codes_path))},
        },
        "countries": countries, "features": features,
    }
    output.write_text(json.dumps(collection, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Wrote {output} ({len(features)} country shapes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cells", type=Path, help="Headerless OWID CSV: x,y,numeric country code")
    parser.add_argument("codes", type=Path, help="Country crosswalk CSV: alpha3,numeric")
    args = parser.parse_args()
    prepare(args.cells, args.codes)
