# Datasheet: Vigil daily burn-day labels, Victoria 2012–13 to 2024–25

Version 1.0. Follows the outline of Gebru et al., "Datasheets for Datasets" (2021).

**Cite as:** Kurra, R. (2026). Vigil daily burn-day labels, Victoria 2012–13 to 2024–25 (v1.0). Zenodo. DOI: [placeholder]

## Motivation

**Why was it made?** To train and test models that predict where a bushfire will burn the next day. No public record gives the area burnt by each fire on each day in Victoria, so this dataset builds one from satellite data.

**Who made it?** Rishwanth Kurra, as part of Vigil, a student portfolio project (data science, RMIT University). No funding and no affiliation with a fire agency.

## Composition

**What is in it?**

- 13 NetCDF files, one per fire season (labelled by the year it ends: 2013 = July 2012 to June 2013, up to 2025). Each covers the same 375 m grid of 2,300 × 1,660 cells in VicGrid94 (EPSG:3111): all of Victoria plus edges of New South Wales and South Australia, inside 140.8° to 150.1° E and 33.8° to 39.3° S. Three variables per cell:
  - `burn_date`: the fire day the cell first burnt (missing if it didn't burn in a modelling event).
  - `event_id`: which fire it belongs to.
  - `uncertain_from`: the date from which the cell's label shouldn't be trusted (missing if never).
- `vigil_events.csv`: one row per fire event, with dates, area, centroid, source, DEECA fire key, detection count and flags.
- `vigil_splits.csv`: each season's cross-validation fold, or test.

**How big is it?** 441 fire events, 278,419 burnt cells, about 3.9 million ha. The 2019–20 Black Summer season holds 52 events and 2.63 million ha of that. 222 of the 441 events burn on fewer than 3 distinct days (flag `under_3_burn_days`).

| Season | Events | Burnt area (kha) | Fold |
|---|---|---|---|
| 2012–13 | 41 | 222.7 | 0 |
| 2013–14 | 40 | 377.4 | 1 |
| 2014–15 | 34 | 36.1 | 2 |
| 2015–16 | 29 | 21.3 | 3 |
| 2016–17 | 30 | 25.4 | 4 |
| 2017–18 | 38 | 64.1 | 3 |
| 2018–19 | 22 | 136.7 | 2 |
| 2019–20 | 52 | 2,628.5 | test |
| 2020–21 | 32 | 26.6 | 2 |
| 2021–22 | 15 | 12.6 | 2 |
| 2022–23 | 33 | 31.1 | 4 |
| 2023–24 | 37 | 76.7 | 3 |
| 2024–25 | 38 | 256.0 | 4 |

**What is an instance?** A fire event: satellite fire detections grouped by being close in space and time, with the burnt cells that belong to it.

**Is it a sample?** It covers the modelling events only: bushfires matched to Victoria's fire records, plus unmatched fires with at least 2 active days and 11 detections, after the exclusions listed below. It is not every fire in the region. Planned burns, small fires, most crop and grass burning, and fires with no burned-area outline are left out.

**What does "fire day" mean?** 06:00 to 06:00 Melbourne local time (Australia/Melbourne: AEST, UTC+10, or AEDT, UTC+11, during daylight saving), named by the date it starts on. Detection times (UTC) are converted with daylight saving applied. A satellite pass at 2 am counts towards the previous day's fire, so documented arrivals land on the right date (for example, the fire reaches Mallacoota on 31 December 2019).

**What do the flags mean?** The `flags` column in `vigil_events.csv` lists, separated by semicolons:

| Flag | Meaning | Events |
|---|---|---|
| `under_3_burn_days` | Its burnt cells fall on fewer than 3 distinct days | 222 |
| `burns_outside_oct_apr` | Some cells burnt outside October to April (the models only use October to April) | 107 |
| `no_scored_days` | No day of this fire was used to train or score the models | 74 |
| `edge_clipped` | Its burned-area patch runs past the edge of the study area, so the fire is cut off | 9 |
| `absorbed_N_fragments` | N small pieces with the same Victorian fire record were merged into it | 8 |

148 events have no flags.

**Is anything missing or unreliable?** Yes, see Limitations. The `uncertain_from` variable marks cells to leave out of scoring. It covers:

- everything within 6 km of the study-area edge or outside it (date 2000-01-02, meaning always);
- fire detections outside any burned-area outline;
- cells belonging to excluded events such as planned burns;
- burned-area patches with no matching detections, so unlabelled burning never counts as "didn't burn".

**Does it contain personal or sensitive data?** No. It describes burnt land at 375 m.

## How it was made

1. **Detections.** NASA FIRMS VIIRS S-NPP 375 m active-fire detections (standard processing), 2012 onwards. NOAA-20 is left out so detection rates are the same across all years.
   - Removed: non-vegetation detections (`type` ≠ 0) and low-confidence detections.
   - Also removed: persistent hot spots (industrial sites, a cell with 10 or more days across 5 or more seasons). The exception is any detection inside a same-season Victorian bushfire outline.
   - This left 630,933 detections.
2. **Events.** Detections within 5 km and 14 days of each other are linked into one event. The 14-day gap bridges smoke and cloud that hid the 2019–20 fires for over a week.
   - An event is tagged as a Victorian bushfire if at least 37.5% of its detections fall inside DEECA bushfire outlines.
   - Small fragments that share a fire record and burned-area patch with a larger event are merged into it.
3. **Exclusions.** These events are dropped:
   - planned burns;
   - small unmatched events;
   - unmatched events mostly on cropland (stubble burns, using the CSIRO fuel map);
   - stop-start unmatched events (3 or more bursts with no burst holding a majority);
   - events with no or little MODIS burned area;
   - events that span seasons in different folds;
   - one manually reviewed chain of autumn burn-offs in western Victoria in 2021.
4. **Burnt area and timing.** The burnt area is NASA MODIS MCD64A1 Collection 6.1 burned area (500 m, monthly), as connected patches. This follows Parks (2014).
   - Each burnt cell gets the date of the nearest VIIRS detection dated within 2 days before to 3 days after the cell's MODIS burn date. If there is none, the nearest detection is used.
   - The cell's event is that detection's event.

## Quality checks

These were computed on the pipeline's labels (`labels.nc` in the repository). The release files were checked against those labels and match cell for cell in every season (burn date, event and uncertainty).

- **Timing:** on 2019–20 cells where the true detection was hidden, the estimated burn day had a mean error of 0.9 days, and 91% were within 1 day.
- **Extent, MODIS against Victorian records, 2019–20 inside Victoria:** IoU 0.73. MODIS catches 81% of the DEECA area, and 88% of the MODIS area lies inside DEECA outlines.
- **Detections alone:** they cover only 49% of DEECA's 2019–20 burnt area, which is why the MODIS outlines are used.

## Limitations

- **Timing is accurate to about a day.** Smoke and cloud can hide fires from the satellite for several days.
- **Grass and crop fires are under-represented.** MODIS misses most grass and crop burning, and cropland-dominated unmatched events are dropped. Real Victorian crop and grass fires (matched to state records) are kept, but NSW and SA ones are always unmatched and so dropped.
- **Edge effects.** The input data stops at the study area's box, so fires crossing it are cut off (three large NSW fires in 2019–20: Batemans Bay and two near Braidwood). Everything within 6 km of the edge is marked uncertain.
- **Fire-season scope.** The models only used October to April. Labels outside those months are included but weren't checked as closely.
- **Unmatched events can still contain non-bushfire burning** that the filters missed.
- **Linking can merge separate fires.** Detections within 5 km and 14 days are joined, so two nearby fires burning at the same time, or a later burn next to an earlier fire, can end up as one event. Known cases include Victorian fire records for complexes of separate ignitions 11 to 37 km apart, and a 2021 chain of autumn burn-offs in western Victoria (excluded manually).
- **Burn days come from MODIS outlines plus VIIRS timing**, so small or fast fires below 500 m may be missing.

## Benchmark

Next-day prediction on the 2019–20 test season (39 scored fires, evaluated once, with a protocol fixed in advance). Metric: F1 on cells newly burnt the next day, pooled over all test fire-days, leaving out cells already burnt and cells marked by `uncertain_from`. F1 balances catching the cells that burnt against flagging cells that didn't (0 to 1). 95% ranges come from resampling the test fires 2,000 times; neural network scores average 3 training runs.

| Model | Test F1 | 95% range |
|---|---|---|
| Main model (U-Net given a physics prediction, no raw weather) | 0.362 | 0.330 to 0.428 |
| Plain U-Net | 0.311 | 0.266 to 0.366 |
| Wind-shifted persistence (today's active fire, pushed downwind) | 0.265 | 0.252 to 0.307 |
| Physics model alone (CSIRO spread equations) | 0.085 | 0.043 to 0.138 |

Details, including scores by distance from the fire and where the models fail: https://vigil-c5o.pages.dev/findings

## Uses

**Intended:** training and evaluating next-day fire-spread models; studying how fires grew day by day.

**Used for:** Vigil's models (https://vigil-c5o.pages.dev). The 2019–20 season was the held-out test set, run once.

**Splits:** `vigil_splits.csv` gives the 5-fold cross-validation over whole seasons, plus 2019–20 as test. Keep seasons whole: fires in one season share weather, so splitting by day or by fire within a season leaks.

**How seasons were assigned to folds:** 2019–20 was fixed as the test season in advance (Black Summer, the most extreme season). The other 12 seasons were split into 5 folds of whole seasons by an exhaustive search for the most even number of training windows with new burn per fold (217 to 245), with the constraint that the three largest, most test-like seasons (2012–13, 2013–14 and 2018–19) sit in different folds. Events that span two seasons in different folds were dropped.

**Not for:** safety decisions, operational fire prediction, or fire-risk ratings. The dataset describes past fires only. For current fires in Victoria, see VicEmergency (https://emergency.vic.gov.au).

## Distribution

- **Licence:** CC BY 4.0. Please cite this dataset and the sources below.
- **Code that builds it:** https://github.com/Se4-gul/vigil (`src/vigil/events.py`, `src/vigil/labels.py`, `src/vigil/release.py`). The raw detections are not included; the download scripts fetch them.

## Sources and credits

- NASA FIRMS VIIRS S-NPP 375 m active fire detections (NASA open data).
- NASA MODIS MCD64A1 Collection 6.1 burned area, LP DAAC, https://doi.org/10.5067/MODIS/MCD64A1.061 (NASA open data).
- State of Victoria (DEECA) fire history, CC BY 4.0. Used to tag events and to exempt detections from the hot-spot filter.
- CSIRO Bushfire Fuel Classification fuel types map (Joshi et al. 2025), https://doi.org/10.25919/vnma-0j64, CC BY-NC 4.0. Used only for one filtering decision: dropping unmatched events with most detections on cropland. No fuel types, fuel classes or other values from the map are included in this dataset.
- Method: Parks, S. A. (2014). Mapping day-of-burning with coarse-resolution satellite fire-detection data. *International Journal of Wildland Fire*, 23(2), 215–223. https://doi.org/10.1071/WF13138

## Maintenance

Maintained by Rishwanth Kurra. Report problems as issues on the GitHub repository. Rebuild with `python -m vigil.release` from the repository after running the pipeline. File checksums are in `SHA256SUMS`.
