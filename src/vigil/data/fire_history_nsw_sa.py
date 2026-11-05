"""Download NSW and SA fire history (burnt-area polygons) for time since burnt.

Sources:
  NSW  NPWS Fire History - Wildfires and Prescribed Burns (SEED, DCCEEW NSW),
       from 1920, updated monthly. Mainly national parks; RFS fires added at
       times, so private land is patchy. Licence CC BY.
  SA   DEW Bushfires and Prescribed Burns History (data.sa.gov.au). Major
       bushfires and planned burns. Last updated 2020-02, so it has nothing
       after early 2020. Licence CC BY.

Output: fire_history_nsw/  and  fire_history_sa/  (unzipped)

Usage:
  python -m vigil.data.fire_history_nsw_sa
"""

import urllib.request
import zipfile
from pathlib import Path

from vigil.config import raw_dir

SOURCES = {
    "fire_history_nsw": "https://datasets.seed.nsw.gov.au/dataset/1d05e145-80cb-4275-af9b-327a1536798d/resource/3853d6c0-6517-4cba-8482-68f93a6d43b1/download/fire_npwsfirehistory.zip",
    "fire_history_sa": "https://www.waterconnect.sa.gov.au/Content/Downloads/DEWNR/FIREMGT_FireHistory_shp.zip",
}


def safe_extract(zip_path: Path, out_dir: Path) -> None:
    """Unzip, refusing members that would land outside out_dir."""
    with zipfile.ZipFile(zip_path) as z:
        for m in z.namelist():
            target = (out_dir / m).resolve()
            if not str(target).startswith(str(out_dir.resolve())):
                raise ValueError(f"unsafe path in zip: {m}")
        z.extractall(out_dir)


def main():
    for name, url in SOURCES.items():
        out = raw_dir(name)
        zip_path = out / "download.zip"
        if zip_path.exists():
            print(f"  {name}: already downloaded")
            continue
        tmp = zip_path.with_suffix(".tmp")
        req = urllib.request.Request(url, headers={"User-Agent": "vigil-research/0.1"})
        with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
            f.write(r.read())
        tmp.rename(zip_path)  # only keep complete files
        safe_extract(zip_path, out)
        print(f"  {name}: {zip_path.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
