# Data

Raw downloads go in `data/raw/` and are not committed to git. Every source is free.

| Folder | What | Source | Grid | Account? | Script |
|---|---|---|---|---|---|
| `fire_history/` | Bushfires and planned burns since 1980 | DEECA Fire History, via WFS | polygons | No | `python -m vigil.data.fire_history` |
| `silo/` | Daily max/min temp, humidity at max temp, rain, vapour pressure deficit | SILO (Qld Government) | 0.05° (~5 km) | No | `python -m vigil.data.silo` |
| `dem/` | Elevation | Copernicus DEM GLO-90 | 90 m | No | `python -m vigil.data.dem` |

## Notes

- **Fire history** has `firetype` = `Bushfire` or `Burn` (planned burn), so labels don't mix the two. CFA fires on private land are included from 2009 on, so there are many more small fires recorded after 2009. Keep this in mind before treating "more fires in recent years" as a real trend.
- For the spread replay of a single day, Open-Meteo's historical API (no key) works for point wind at a few locations.

## Attribution

- Fire history: © State of Victoria (DEECA), CC BY 4.0
- SILO: © State of Queensland, CC BY 4.0
- Copernicus DEM: © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018, provided under COPERNICUS by the European Union and ESA
