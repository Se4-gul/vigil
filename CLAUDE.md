# Vigil

Bushfire spread prediction for Victoria, with a public interactive map. Portfolio project by Rishi (first-year data science, RMIT).

## How to work with Rishi
- Claude writes the code. Rishi makes the design decisions and must be able to defend them in an interview.
- At each real design choice: explain the options in a few lines, give a recommendation, and let Rishi decide. Record the decision in the Decisions section below.
- Keep chat replies short. Rishi doesn't read long docs or READMEs, so put the important part first.
- Give genuine pushback. Don't just agree.
- Plumbing (downloads, glue, config) needs no discussion. Just do it well.

## Plan
1. **Data pipeline:** group NASA FIRMS VIIRS 375 m detections (2012 onwards) into fire events and build daily burnt-area maps. Align inputs: BARRA2 weather and wind, DEM (slope, aspect), fuel type, DEA fuel moisture.
2. **Physics baseline:** a cellular automaton using CSIRO spread models (dry eucalypt forest, grassland).
3. **ML model:** a U-Net predicting the next-day fire map (following Huot et al. 2022, "Next Day Wildfire Spread"). Train 2012–2018; hold out Black Summer 2019–20 as the test set.
4. **Compare:** physics vs ML vs hybrid (physics output as an ML input). Metrics: IoU, precision/recall/F1 on newly burnt cells.
5. **Website:** a map where you click to start a hypothetical fire (pick the weather), or replay a real fire, and compare model predictions with what actually burnt.

## Data
| Source | Status | Notes |
|---|---|---|
| Copernicus DEM 90 m (`data/raw/dem`) | done, 68 tiles | 3 arc-second cells, about 74 m east-west near 37°S. Work out slope using the real east-west width per row. |
| SILO daily weather (`data/raw/silo`) | done, 5 vars × 2000–2025 | 5 km. Possibly redundant with BARRA2. |
| DEECA fire history (`data/raw/fire_history`) | done, 414k records, ~10k large-fire outlines | Multiple rows per firekey; ~57% of start dates fall on the 1st of the month (placeholders?); bbox centre isn't the ignition point; firetype includes "Other". Use for final burnt areas and checking, not exact timing. |
| ERA5 wind (`data/raw/era5`) | 21/26 years (missing 2011–2015) | Superseded by BARRA2; don't extend. |
| NASA FIRMS VIIRS 375 m | to do | firms.modaps.eosdis.nasa.gov/download, needs Earthdata login. 2012 onwards. |
| BARRA2 | to do | s3://bom-opendata-climate/BARRA2/ (ap-southeast-2), no account needed. BARRA-C2 is the km-scale version. |
| DEA Fuel Moisture Content | to do | 20 m, Sentinel-2, 2015 onwards. |
| Victoria fuel type layer | to find | source not yet confirmed |

Download scripts are in `src/vigil/data/`; settings are in `src/vigil/config.py`. Write downloads to a `.tmp` file and rename when complete, so partial files never look finished.

## Decisions
- 2026-11-05: Dropped the ignition risk model. Spread prediction is the headline.
- 2026-11-05: BARRA2 replaces ERA5 (finer resolution, no queue).
- Earlier, from the benched learning build, still relevant: work out terrain features at full DEM resolution, then summarise to the coarser grid (never average elevation first). Summarise slope as the fraction of each cell above slope thresholds, with thresholds picked from a histogram.

## Open decisions
- Spread grid resolution (375 m matches VIIRS)
- Fuel type source
- Website stack
