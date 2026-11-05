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
3. ML model, trained on 2012–2018
4. Compare physics vs ML vs hybrid on Black Summer 2019–20 (held out)
5. Website: start a hypothetical fire or replay a real one, and compare predictions with what actually burnt

## Evaluation

IoU, plus precision, recall and F1 on newly burnt cells, with the 2019–20 season held out as the test set.

## Known limits

- Recorded weather is used instead of historical forecasts, so real-world accuracy would be lower.
- No firefighting is modelled.
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
