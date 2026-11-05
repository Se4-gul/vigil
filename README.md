# Vigil

Bushfire spread prediction for Victoria, with a public interactive map.

## What it does

Predicts where a fire will have burnt by the next day, using two approaches and a combination of both:

1. **Physics baseline:** a cellular automaton using CSIRO spread models (dry eucalypt forest, grassland).
2. **ML model:** a U-Net that predicts the next day's fire map from today's fire, weather, terrain and fuel (following Huot et al. 2022, "Next Day Wildfire Spread").
3. **Hybrid:** the physics prediction fed in as an extra input to the ML model.

## Plan

1. Data pipeline: group NASA FIRMS VIIRS 375 m detections (2012 onwards) into fire events and build daily burnt-area maps, aligned with BARRA2 weather, terrain, fuel type and fuel moisture.
2. Physics baseline
3. ML model: model selection by leave-one-season-out cross-validation over 2013–2019 and 2021–2025; final model trained on all of them
4. Compare physics vs ML vs hybrid on Black Summer (season 2020, held out)
5. Website: start a hypothetical fire or replay a real one, and compare predictions with what actually burnt

## Evaluation

IoU, plus precision, recall and F1 on newly burnt cells. Test: season 2020 (Black Summer). Model selection: leave-one-season-out cross-validation over the other 12 seasons. Seasons after the test are used for training on purpose: spread physics does not change between years, and fire data is scarce.

## Scope

Fire season only: predictions are made for days in October–April (all splits, including the test set). Weather data was prepared for those months, and fires outside them are mostly off-season burns.

## Known limits

- Recorded weather (BARRA reanalysis) is used for the day being predicted, which assumes a perfect forecast; real-world accuracy with forecasts would be lower.
- No firefighting is modelled.
- The ML model has seen very few grass and crop fires (stubble burns are filtered out, and the MODIS burned-area outlines used for labels miss most grass burning). The website uses the physics model, or shows a warning, for grassland ignitions.
- Black Summer had extreme fire behaviour (pyroconvection, long-range spotting) rarely seen in the training seasons, so the test checks how well the model holds up on conditions more extreme than it trained on.
- Satellite detections miss fire under cloud and smoke, and VIIRS passes only a few times a day.

## Layout

```
data/raw/        downloaded source data (not committed)
data/processed/  gridded, joined data (not committed)
notebooks/       exploration
src/vigil/       data/ and spread/ code
app/             website
docs/            write-up and figures
```
