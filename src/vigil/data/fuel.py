"""Download the CSIRO national fuel type map and crop it to our bounds.

Source: CSIRO Bushfire Fuel Classification (BFC) fuel types map, release 1.0
(Joshi et al. 2025, IJWF 34 WF24224; DOI 10.25919/vnma-0j64), from the CSIRO
Data Access Portal. Licence CC BY-NC 4.0 (non-commercial use only).

National, so it covers the NSW and SA parts of our bounds the same way as
Victoria. Static: one fuel picture, with no record of recent fires (see the
time-since-burnt feature for that).

Output: fuel/bfc_fuel_types.tif            national map (~220 MB)
        fuel/bfc_attribute_table.pdf       class codes and names
        fuel/bfc_metadata.pdf
        fuel/bfc_fuel_types_crop.tif       cropped to config bounds

Usage:
  python -m vigil.data.fuel
"""

import urllib.request

import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

from vigil.config import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, raw_dir

DAP = "https://data.csiro.au/dap/ws/v2/collections/65550/data/"
FILES = {
    "bfc_fuel_types.tif": "82997072",
    "bfc_attribute_table.pdf": "82997074",
    "bfc_metadata.pdf": "82997073",
}


def download(name: str, file_id: str) -> None:
    path = raw_dir("fuel") / name
    if path.exists():
        print(f"  {name}: already downloaded")
        return
    tmp = path.with_suffix(".tmp")
    urllib.request.urlretrieve(DAP + file_id, tmp)
    tmp.rename(path)  # only keep complete files
    print(f"  {name}: {path.stat().st_size / 1e6:.0f} MB")


def crop() -> None:
    src_path = raw_dir("fuel") / "bfc_fuel_types.tif"
    out = raw_dir("fuel") / "bfc_fuel_types_crop.tif"
    with rasterio.open(src_path) as src:
        bounds = transform_bounds("EPSG:4326", src.crs, LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
        window = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
        data = src.read(1, window=window)
        profile = src.profile | {
            "height": data.shape[0], "width": data.shape[1],
            "transform": src.window_transform(window), "compress": "deflate",
        }
        tmp = out.with_suffix(".tmp")
        with rasterio.open(tmp, "w", driver="GTiff", **{k: v for k, v in profile.items()
                                                       if k != "driver"}) as dst:
            dst.write(data, 1)
            try:
                dst.write_colormap(1, src.colormap(1))
            except ValueError:  # no colour table in the source
                pass
        tmp.rename(out)
        print(f"  cropped: {data.shape[1]} x {data.shape[0]} cells, CRS {src.crs}, "
              f"cell {src.res[0]} x {src.res[1]} {src.crs.linear_units}")


def main():
    for name, file_id in FILES.items():
        download(name, file_id)
    crop()


if __name__ == "__main__":
    main()
