# Europe choropleth

Make PNG maps from CSV data with Natural Earth boundaries or an OWID cartogram.
Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

## Try the example

```sh
uv run make_map.py
uv run make_map.py --geometry owid
```

PNGs go in `output/`. Both map layouts and the font are bundled.

## Use your data

Save `values.csv`:

```csv
code,value,Men,Women
BEL,33.4,35.2,31.5
DEU,39.5,41.6,37.4
```

Save `map.json` beside it:

```json
{
  "data": "values.csv",
  "title": "Expected career duration (years)",
  "source": "OECD, Pensions at a Glance 2025",
  "geometry": "natural-earth",
  "scale": {"method": "quantile", "reverse": false},
  "key": {
    "columns": [{"column": "Men"}, {"column": "Women"}]
  }
}
```

```sh
uv run make_map.py map.json --output output/map.png
```

Use ISO alpha-2 or alpha-3 country codes. The code column is detected from its values.
Unmatched codes produce warnings and their rows are skipped. Countries within the European view are mapped.
Empty values appear grey. CSV paths are relative to the metadata file.
`key.columns` adds fields to the country key. Headings use the CSV column names.

- Set `geometry` to `owid` for the cartogram.
- Choose `linear` or `quantile` (continuous ECDF). Set `reverse` to `true` to flip Spectral.
- Set `value` to the numeric CSV column name; the default is `value`.
- Set captions with `subtitle` and `note`; adjust size with `dpi` and `font_scale`.

See the [example metadata](data/expected_career_duration_oecd_mean.json) for column formatting.

## Credits

[Natural Earth](https://www.naturalearthdata.com/) is public domain.
[OWID cartogram](https://ourworldindata.org/world-population-cartogram): [MIT](data/owid/LICENSE).
[Spleen 2.2.0](https://github.com/fcambus/spleen/releases/tag/2.2.0), Frederic Cambus: [BSD-2-Clause](fonts/spleen/LICENSE).
Example data: [OECD, Pensions at a Glance 2025](https://www.oecd.org/en/publications/pensions-at-a-glance-2025_e40274c1-en.html).

Run tests with `uv run python -m unittest discover -s tests`.
