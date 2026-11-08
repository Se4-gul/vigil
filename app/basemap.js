// Shared basemap for every map page: OpenFreeMap "dark" (vector tiles, free,
// no API key), lifted a step (water darkest, land a notch lighter, roads and
// labels dimmed), with a very subtle AWS Terrarium hillshade. Calls onReady(map,
// firstLabelLayer) once the style is loaded; fire layers go under the labels.
function makeBasemap(onReady, view = { center: [148.6, -37.3], zoom: 7 },
                     { container = "map", scale = true, attribution = true } = {}) {
  const map = new maplibregl.Map({
    container, attributionControl: attribution ? { compact: true } : false,
    style: "https://tiles.openfreemap.org/styles/dark", maxZoom: 12, ...view,
  });
  // The OpenFreeMap dark style names a few icons its own sprite lacks (e.g.
  // "circle-11"); give them a transparent pixel instead of a console warning.
  map.on("styleimagemissing", e => {
    if (!map.hasImage(e.id)) map.addImage(e.id, { width: 1, height: 1, data: new Uint8Array(4) });
  });
  if (scale) map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-right");
  map.on("style.load", () => {
    let firstLabelLayer;
    const set = (id, prop, v) => map.getLayer(id) && map.setPaintProperty(id, prop, v);
    set("background", "background-color", "#26292d");
    set("water", "fill-color", "#0f1215");
    set("waterway", "line-color", "#0f1215");
    for (const id of ["landcover_wood", "landuse_park"]) set(id, "fill-color", "#2a2d31");
    set("landuse_residential", "fill-color", "#2c2f33");
    for (const layer of map.getStyle().layers) {
      if (layer.type === "symbol" && !firstLabelLayer) firstLabelLayer = layer.id;
      if (layer.type === "line" && layer["source-layer"] === "transportation") set(layer.id, "line-opacity", 0.4);
      if (layer.type === "line" && layer["source-layer"] === "boundary") set(layer.id, "line-opacity", 0.55);
      if (layer.type === "symbol" && layer.layout && layer.layout["text-field"]) set(layer.id, "text-opacity", 0.5);
    }
    // The virtual-host URL is the one that sends CORS headers.
    map.addSource("dem", {
      type: "raster-dem", encoding: "terrarium", tileSize: 256, maxzoom: 12,
      tiles: ["https://elevation-tiles-prod.s3.amazonaws.com/terrarium/{z}/{x}/{y}.png"],
      attribution: 'Terrain: <a href="https://registry.opendata.aws/terrain-tiles/">Terrain Tiles (AWS Open Data, Mapzen/Tilezen)</a>',
    });
    const firstLine = map.getStyle().layers.find(l => l.type === "line").id;
    map.addLayer({ id: "hillshade", type: "hillshade", source: "dem",
                   paint: { "hillshade-exaggeration": 0.32, "hillshade-shadow-color": "#0b0d0f",
                            "hillshade-highlight-color": "#4c4f54", "hillshade-accent-color": "#0b0d0f" } },
                 firstLine);
    onReady(map, firstLabelLayer);
  });
  return map;
}

// The clear part of the map: below the title block and above the readout,
// measured from the layout so the fire never sits under the text.
function clearArea(topSel = ".top", bottomSel = ".readout") {
  const vh = innerHeight, vw = innerWidth, gap = 24;
  const top = document.querySelector(topSel).getBoundingClientRect().bottom + gap;
  const bottom = vh - document.querySelector(bottomSel).getBoundingClientRect().top + gap;
  const side = Math.min(80, vw * 0.06);
  const spare = vh - top - bottom;                 // keep at least a third of the height for the fire
  const shrink = spare < vh / 3 ? (vh / 3 - spare) / 2 : 0;
  return { top: Math.max(top - shrink, 10), bottom: Math.max(bottom - shrink, 10), left: side, right: side };
}
