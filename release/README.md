# Vigil fire labels

Day-by-day burnt area of 441 bushfires in Victoria (and edges of NSW and SA), fire seasons 2012–13 to 2024–25, on a 375 m grid. Built from NASA VIIRS fire detections and MODIS burned area to train and test next-day fire-spread models.

Licence: CC BY 4.0. Read [DATASHEET.md](DATASHEET.md) before using it: it covers how the labels were made, what they miss, and the flags.

## Files

| File | What |
|---|---|
| `data/vigil_labels_<season>.nc` | One per season (2013 = 2012–13 ... 2025 = 2024–25). CF-1.8 NetCDF, 2,300 × 1,660 cells, 375 m, EPSG:3111 (VicGrid94). `burn_date` (fire day the cell first burnt), `event_id`, `uncertain_from` (date from which to leave the cell out of scoring). Missing values are -1 on disk and decode to NaN/NaT. |
| `vigil_events.csv` | One row per event: season, fold, source, DEECA fire key, first and last burn date, burn days, cells, area (ha), centroid (lon, lat), detections, flags. |
| `vigil_splits.csv` | Season to cross-validation fold (0 to 4) or test. Seasons are whole; 2019–20 is the test season and was evaluated once. |
| `load_example.py` | Open a season, list its biggest fires, count one day's newly burnt cells. |
| `SHA256SUMS` | Checksums for the data files. |

A fire day runs 06:00 to 06:00 local time and is named by the date it starts on. When scoring a prediction for fire day T, leave out cells where `uncertain_from <= T`.

## Quick start

```
pip install xarray h5netcdf pandas pyproj
python load_example.py
```

## Rebuild

From the repository (https://github.com/Se4-gul/vigil), after running the pipeline up to the labels: `python -m vigil.release`.

## Cite

Kurra, R. (2026). Vigil daily burn-day labels, Victoria 2012–13 to 2024–25 (version 1.0) [Data set]. Licence CC BY 4.0.
Please also cite the sources listed in the datasheet, and Parks (2014) for the method.

Not for safety decisions. For current fires, see VicEmergency (https://emergency.vic.gov.au).
