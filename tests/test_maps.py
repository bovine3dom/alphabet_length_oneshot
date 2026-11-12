import io
import json
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stderr

import numpy as np
from PIL import Image
from shapely.geometry import LineString, box, shape

import make_map as maps
import prepare_cartogram
from prepare_geography import clip_country


class MapTests(unittest.TestCase):
    def setUp(self):
        self.stderr = self.enterContext(redirect_stderr(io.StringIO()))
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.csv = self.root / "values.csv"
        self.csv.write_text("country,score,Men\nBE,33.4,35.2\nFRA,34.9,36.3\nNOR,,42.2\nISR,39,40.9\nOECD,37.3,40.2\nZZZ,1,2\n")
        self.metadata = self.root / "map.json"
        self.metadata.write_text(json.dumps({
            "data": "values.csv", "title": "Career duration", "source": "Example data",
            "value": "score",
            "key": {"columns": [{"column": "Men", "format": "%.1f"}]},
        }))

    def test_csv_country_codes_and_missing_values(self):
        _, _, values, _, records = maps.load_map(self.metadata)
        self.assertEqual(values, {"BEL": 33.4, "FRA": 34.9})
        self.assertEqual(records["BEL"]["Men"], "35.2")
        self.assertIn("OECD", self.stderr.getvalue())
        self.assertIn("ZZZ", self.stderr.getvalue())
        self.csv.write_text(self.csv.read_text().replace("country,", "identifier,", 1))
        self.assertEqual(maps.load_map(self.metadata)[2], values)

    def test_invalid_csv(self):
        for body in ("BE,1,2\nBEL,2,3\n", "BE,inf,2\n", "BE,words,2\n"):
            with self.subTest(body=body), self.assertRaises(ValueError):
                self.csv.write_text("country,score,Men\n" + body)
                maps.load_map(self.metadata)
        self.csv.write_text("code,identifier,score,Men\nBE,FR,1,2\n")
        with self.assertRaisesRegex(ValueError, "Multiple country-code columns"):
            maps.load_map(self.metadata)

    def test_continuous_scales(self):
        linear = maps.make_norm([0, 10, 100], {"method": "linear"})
        np.testing.assert_allclose(linear([0, 10, 100]), [0, 0.1, 1])
        quantile = maps.make_norm([0, 0, 10, 100], {"method": "quantile"})
        values = [0, 5, 10, 55, 100]
        np.testing.assert_allclose(quantile(values), [0, 0.3, 0.6, 0.8, 1])
        np.testing.assert_allclose(quantile.inverse(quantile(values)), values)
        self.assertEqual(maps.make_norm([7, 7], {})(7), 0.5)
        np.testing.assert_allclose(maps.spectral({"reverse": True})(0.0), maps.CMAP(1.0))

    def test_geometry_modes_share_observations(self):
        natural = maps.load_map(self.metadata)
        cartogram = maps.load_map(self.metadata, "owid")
        self.assertEqual(natural[2], cartogram[2])
        np.testing.assert_allclose(natural[3](list(natural[2].values())),
                                   cartogram[3](list(cartogram[2].values())))

    def test_crop_preserves_real_borders(self):
        view = box(-1, -1, 1, 1)
        fill, boundary = clip_country(box(-2, -2, 0.5, 2), view)
        self.assertGreater(fill.area, 0)
        self.assertTrue(boundary.equals(LineString([(0.5, -1), (0.5, 1)])))
        _, boundary = clip_country(box(-1, -2, 0.5, 2), view)
        self.assertTrue(boundary.covers(LineString([(-1, -1), (-1, 1)])))

    def test_cartogram_import(self):
        cells, codes, output = [self.root / name for name in ("cells.csv", "codes.csv", "map.geojson")]
        cells.write_text("400,200,250\n402,200,250\n440,200,56\n")
        codes.write_text("alpha3,numeric\nFRA,250\nBEL,056\n")
        prepare_cartogram.prepare(cells, codes, output)
        features = {f["properties"]["ADM0_A3"]: f for f in json.loads(output.read_text())["features"]}
        self.assertEqual(set(features), {"FRA", "BEL"})
        self.assertEqual(shape(features["FRA"]["geometry"]).area, 8)
        self.assertLess(features["FRA"]["properties"]["LABEL_Y"], 0)

    def test_cartogram_pixel_alignment(self):
        fig, ax = maps.plt.subplots()
        try:
            ax.set_xlim(340, 620)
            ax.set_ylim(-330, -100)
            maps.fit_cartogram_pixels(ax, prepare_cartogram.BOUNDS, maps.DPI)
            pixels = np.array(ax.get_window_extent().bounds) * maps.DPI / fig.dpi
            np.testing.assert_allclose(pixels, np.rint(pixels), atol=1e-8)
        finally:
            maps.plt.close(fig)

    def test_both_modes_render_png(self):
        for geometry in maps.GEOMETRIES:
            with self.subTest(geometry=geometry):
                metadata, features, values, norm, records = maps.load_map(self.metadata, geometry)
                output = self.root / f"{geometry}.png"
                maps.render(metadata, features, values, norm, output, records)
                with Image.open(output) as image:
                    self.assertEqual(image.format, "PNG")
                    image.verify()


if __name__ == "__main__":
    unittest.main()
