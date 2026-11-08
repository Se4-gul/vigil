# Data licences

The MIT licence in [LICENSE](LICENSE) covers the code only. The data this project uses keeps
its own licence, and so do files in this repository derived from it (the
images and JSON under app/data/ and app/img/, and the tables in docs/results/
and archive/). Raw data is not included; the download scripts in
src/vigil/data/ fetch it from the sources below.

Because the models and the site use the CSIRO fuel map (CC BY-NC 4.0), the
derived data and the website are for non-commercial use only.

The fire-label dataset in release/ is the exception: it contains no CSIRO
fuel values and is released under CC BY 4.0 (see release/DATASHEET.md).

- NASA FIRMS VIIRS 375 m active fire detections: NASA open data, no
  restrictions on use.
  https://firms.modaps.eosdis.nasa.gov/
  https://www.earthdata.nasa.gov/engage/open-data-services-software-policies/data-use-guidance
- MODIS MCD64A1 Collection 6.1 burned area (Giglio et al.), NASA LP DAAC: NASA
  open data, no restrictions on use.
  https://doi.org/10.5067/MODIS/MCD64A1.061
- Bureau of Meteorology BARRA2 reanalysis (BARRA-C2): CC BY 4.0.
  https://registry.opendata.aws/australian-bureau-of-meteorology-barra-2/
- SILO climate data, Queensland Government: CC BY 4.0.
  https://www.longpaddock.qld.gov.au/silo/
- Copernicus DEM GLO-90: free under the Copernicus DEM licence.
  © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018,
  provided under COPERNICUS by the European Union and ESA.
  https://registry.opendata.aws/copernicus-dem/
- CSIRO Bushfire Fuel Classification fuel types map (Joshi et al. 2025),
  © CSIRO 2025: CC BY-NC 4.0 (non-commercial).
  https://doi.org/10.25919/vnma-0j64
  https://creativecommons.org/licenses/by-nc/4.0/
- Victorian fire history, State of Victoria (DEECA): CC BY 4.0.
  https://discover.data.vic.gov.au/dataset/fire-history-records-of-fires-across-victoria
- NSW fire history, NSW National Parks and Wildlife Service: CC BY 4.0.
  https://datasets.seed.nsw.gov.au/dataset/1d05e145-80cb-4275-af9b-327a1536798d
- South Australian fire history, Department for Environment and Water: CC BY
  (version not stated by the publisher).
  https://data.sa.gov.au/data/dataset/fire-history
- State boundaries (ASGS Edition 3), Australian Bureau of Statistics: CC BY 4.0.
  https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3
- Fonts in app/fonts/ (Newsreader, Barlow Semi Condensed): SIL Open Font
  License 1.1; licence texts alongside the fonts.
- The website's basemap (OpenFreeMap, OpenMapTiles, OpenStreetMap
  contributors, ODbL) and hillshade (AWS Terrain Tiles) are loaded live and
  are not part of this repository.
  https://www.openstreetmap.org/copyright
