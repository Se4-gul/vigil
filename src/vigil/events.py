"""Group VIIRS S-NPP fire detections into fire events.

Steps:
  1. Filter detections: keep vegetation fires (type 0), drop low confidence,
     and drop persistent hot spots (industrial sites, repeat farm burns that
     FIRMS still labels as vegetation).
  2. Link detections that are within DIST_KM of each other and within
     GAP_DAYS in time; each connected group is one event. To keep this fast,
     linking runs on cells of max(1 km, DIST_KM/4) per half-day (day and night
     overpasses), so distances are accurate to about 0.7 of a cell and gaps
     to half a day.
  3. Tag each detection with the DEECA outline it falls in (bushfire or
     planned burn, same season, outline buffered by one VIIRS pixel).

Thresholds are tuned against DEECA bushfire outlines (`tune`): pick the
tightest setting where every DEECA fire keeps at least 95% of its detections
in one event, ignoring pieces more than 30 days from the fire's main event
(later burns inside the outline). Merging is preferred to splitting: merging two fires is harmless
for training (windows are cut around each day's active fire), but splitting
one fire loses its spread history. Not 100%, because DEECA records some
complexes of separate ignitions as one fire (e.g. W202045016).

Output: data/processed/fire_events.parquet, one row per detection.

Usage:
  python -m vigil.events prepare            # filter + tag, cached (~slow, once)
  python -m vigil.events filtercheck        # what the filters removed
  python -m vigil.events tune               # print the threshold grid
  python -m vigil.events build --dist 5 --gap 14   # chosen setting
"""

import argparse
import glob
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from vigil.config import CV_FOLDS, PROCESSED, RAW, fire_day

SEASONS = range(2013, 2026)   # season 2012 dropped (see CLAUDE.md)
VAL_SEASONS, TEST_SEASONS = {2019}, {2020}
CRS = 3111                    # VicGrid94, metres
PIXEL_M = 375

# Persistent hot spot filter: a ~1 km cell seen on at least this many distinct
# days, across at least this many seasons, is industrial or a repeat farm burn.
# 10 days is where the histogram of days per cell drops off. 5 seasons, not 3:
# East Gippsland forest that burnt in both 2014 and 2020 reaches 10+ days over
# 3 seasons. Detections inside a same-season DEECA bushfire are never removed.
STATIC_CELL_DEG = 0.01
STATIC_MIN_DAYS = 10
STATIC_MIN_SEASONS = 5

LINK_CELL_M = 1000
MIN_DETECTIONS_FOR_TUNING = 5  # DEECA fires smaller than this can't really split
SHARE_TARGET = 0.95            # each DEECA fire: this share of detections in one event
MIN_ACTIVE_DAYS = 2            # unmatched events kept for modelling: at least this many days
MIN_UNMATCHED_DETECTIONS = 11  # ... and this many detections
MAX_CROP_SHARE = 0.5           # unmatched events mostly on cropland are stubble burns
CROP_CLASS = 640               # CSIRO BFC "Croplands"
GRASS_CLASSES = [610, 620, 631, 632, 633, 640, 700]   # grassland + crop/horticulture
# An event is a DEECA bushfire only if at least this share of its detections
# fall in DEECA bushfire outlines: the empty bin (0.35-0.40) in the histogram
# of that share over tagged events (455 of 592 are >= 0.95).
DEECA_MIN_SHARE = 0.375
# Stop-start (burn-off chains), unmatched events only: >= BURST_MIN bursts of
# activity separated by quiet gaps > BURST_GAP days, with no burst holding a
# majority of the detections. Real fires seen through smoke have one
# dominant burst (e.g. NSW Black Summer fires: largest burst 66-100%).
BURST_GAP, BURST_MIN, BURST_MAX_TOP = 3, 3, 0.5
# Documented manual exclusions, matched by season, first fire day and location
# (event ids can change between builds). Each must match exactly one event.
MANUAL_EXCLUDE = [
    {"season": 2021, "first_day": "2021-03-02", "lat": -37.83, "lon": 143.28,
     "why": "Western District, 2 Mar-9 May 2021: ~2 months of stop-start bursts on "
            "grassland (pasture), 1,999 detections of which 0.35% touch a DEECA bushfire "
            "outline. Autumn burn-offs chained by the 5 km / 14 d linking, not one bushfire. "
            "Its largest burst holds 52%, just above the stop-start rule's 50%."},
]
# Fragments: small pieces of a DEECA fire that the 5 km / 14 d linking left
# separate (e.g. 8 detections 5.1 km from the Upper Murray fire). Absorbed into
# the larger event that shares their majority DEECA fire key and their
# MCD64A1 patch. Large co-keyed events stay separate (separate ignitions in
# one DEECA complex, e.g. W20141735).
FRAGMENT_MAX_DAYS = 3          # fragment: fewer active fire days than this ...
FRAGMENT_MAX_DETECTIONS = 50   # ... or fewer detections than this
LATE_DAYS = 30                 # pieces this long after/before the main event are later burns, not splits


def fire_season(t: pd.Series) -> pd.Series:
    """Seasons run July-June and are labelled by the year they end."""
    return t.dt.year + (t.dt.month >= 7)


def load_all() -> pd.DataFrame:
    """All S-NPP detections, with `removed_by` naming the first filter that drops
    each one (None = kept). Filters apply in order: type, low_conf, then
    (after DEECA tagging) static, season."""
    files = sorted(glob.glob(str(RAW / "firms" / "VIIRS_SNPP_SP" / "*.csv")))
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d["time"] = pd.to_datetime(d.acq_date) + pd.to_timedelta(
        d.acq_time // 100 * 60 + d.acq_time % 100, unit="min")  # UTC
    d["season"] = fire_season(d.time)

    removed = pd.Series(None, index=d.index, dtype=object)
    removed[d.type != 0] = "type"
    removed[removed.isna() & (d.confidence == "l")] = "low_conf"

    g = gpd.GeoSeries(gpd.points_from_xy(d.longitude, d.latitude), crs=4326).to_crs(CRS)
    d["x"], d["y"] = g.x.values, g.y.values
    d["removed_by"] = removed
    return d


def apply_static_and_season(d: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Persistent hot spots (needs DEECA tags), then the season range."""
    removed = d.removed_by.copy()
    # Judged over the whole record (2012-2025), on detections passing type/conf.
    cell = (d.latitude / STATIC_CELL_DEG).round().astype(int).astype(str) + "_" + \
           (d.longitude / STATIC_CELL_DEG).round().astype(int).astype(str)
    ok = removed.isna()
    stats = d[ok].groupby(cell[ok]).agg(days=("acq_date", "nunique"),
                                        seasons=("season", "nunique"))
    static = stats.index[(stats.days >= STATIC_MIN_DAYS) & (stats.seasons >= STATIC_MIN_SEASONS)]
    in_static = ok & cell.isin(static)
    exempt = in_static & d.deeca_type.eq("Bushfire")
    removed[in_static & ~exempt] = "static"
    removed[removed.isna() & ~d.season.isin(SEASONS)] = "season"
    d["removed_by"] = removed
    d["static_cell"] = cell.where(cell.isin(static))

    if verbose:
        print(f"S-NPP detections: {len(d):,}")
        for f, label in [("type", "type != 0 (non-vegetation)"), ("low_conf", "low confidence"),
                         ("static", f"persistent hot spots ({len(static)} cells)"),
                         ("season", "outside seasons 2013-2025")]:
            print(f"  {label:<38} -{(removed == f).sum():,}")
        print(f"  (static cells kept because inside a DEECA bushfire: {exempt.sum():,})")
        print(f"  kept: {removed.isna().sum():,}")
    return d


def tag_deeca(d: pd.DataFrame) -> pd.DataFrame:
    """Add deeca_type (Bushfire/Burn/None) and deeca_firekey: the same-season
    DEECA outline each detection falls in, widened by one VIIRS pixel."""
    feats = [json.loads(line) for line in open(RAW / "fire_history" / "large_fires.geojsonl")]
    p = gpd.GeoDataFrame.from_features(feats, crs=4326)
    p = p[p.season.isin(SEASONS) & p.firetype.isin(["Bushfire", "Burn"])]
    p = p[["firetype", "firekey", "season", "geometry"]].to_crs(CRS)
    p["geometry"] = p.geometry.make_valid()
    p["firekey"] = p.firekey.str.strip().replace("", None)

    d["deeca_type"] = None
    d["deeca_firekey"] = None
    for season, ds in d[d.season.isin(SEASONS)].groupby("season"):
        ps = p[p.season == season]
        pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(ds.x, ds.y), index=ds.index, crs=CRS)
        j = gpd.sjoin(pts, ps, predicate="dwithin", distance=PIXEL_M, how="inner")
        # Bushfire wins over Burn if outlines overlap; keep one match per detection.
        j = j.assign(rank=(j.firetype != "Bushfire").astype(int)).sort_values("rank")
        j = j[~j.index.duplicated()]
        d.loc[j.index, "deeca_type"] = j.firetype
        d.loc[j.index, "deeca_firekey"] = j.firekey.where(j.firetype == "Bushfire")
    return d


CACHE = PROCESSED / "detections_tagged.parquet"


def prepare() -> pd.DataFrame:
    """Filter and DEECA-tag every detection once; later steps read the cache."""
    d = apply_static_and_season(tag_deeca(load_all()))
    PROCESSED.mkdir(parents=True, exist_ok=True)
    tmp = CACHE.with_suffix(".tmp")
    d.drop(columns=["acq_date", "acq_time"]).to_parquet(tmp, index=False)
    tmp.rename(CACHE)
    return d


def load_detections() -> pd.DataFrame:
    """Kept detections (all filters passed), with DEECA tags."""
    d = pd.read_parquet(CACHE)
    return d[d.removed_by.isna()].reset_index(drop=True)


def filter_check(d: pd.DataFrame) -> None:
    """How many removed detections sit inside a same-season DEECA bushfire."""
    r = d[d.removed_by.notna() & d.season.isin(SEASONS)].copy()
    r["in_bushfire"] = r.deeca_type.eq("Bushfire")
    print("\nRemoved detections inside DEECA bushfire outlines, by filter:")
    print(r.groupby("removed_by").in_bushfire.agg(removed="size", in_bushfire="sum").to_string())
    t = r[r.in_bushfire].pivot_table(index="season", columns="removed_by", values="x",
                                     aggfunc="size", fill_value=0)
    print("\nBy season (removed and inside a DEECA bushfire):")
    print(t.to_string())
    kept = d[d.removed_by.isna() & d.deeca_type.eq("Bushfire")].groupby("season").size()
    low = t.get("low_conf", pd.Series(0, index=t.index))
    print("\nLow-confidence removed as a share of all bushfire detections, by season:")
    print((low / (low + kept.reindex(low.index).fillna(0))).round(3).to_string())


def link(d: pd.DataFrame, dist_km: float, gap_days: float) -> np.ndarray:
    """Event id per detection: connected components of the space-time links."""
    dist_m = dist_km * 1000
    cell = max(LINK_CELL_M, dist_m / 4)
    # Collapse to one node per cell per half-day.
    t_days = (d.time - pd.Timestamp("2000-01-01")).dt.total_seconds().values / 86400
    cx = np.floor(d.x.values / cell)
    cy = np.floor(d.y.values / cell)
    tb = np.floor(t_days * 2)
    keys, node = np.unique(np.stack([cx, cy, tb], axis=1), axis=0, return_inverse=True)
    node = node.ravel()
    nx = (keys[:, 0] + 0.5) * cell
    ny = (keys[:, 1] + 0.5) * cell
    nt = keys[:, 2] / 2

    # Box neighbours in scaled space, then keep pairs within the true circle.
    tree = cKDTree(np.stack([nx / dist_m, ny / dist_m, nt / gap_days], axis=1))
    pairs = tree.query_pairs(1.0, p=np.inf, output_type="ndarray")
    if len(pairs):
        dx = nx[pairs[:, 0]] - nx[pairs[:, 1]]
        dy = ny[pairs[:, 0]] - ny[pairs[:, 1]]
        pairs = pairs[dx * dx + dy * dy <= dist_m * dist_m]
    n = len(keys)
    graph = coo_matrix((np.ones(len(pairs), dtype=bool), (pairs[:, 0], pairs[:, 1])),
                       shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return labels[node]


def split_late(d: pd.DataFrame, event: np.ndarray, dist_km: float, gap_days: float,
               max_rounds: int = 5) -> np.ndarray:
    """Detach detections more than LATE_DAYS outside an event's main activity
    (the span holding its middle 95% of detections) and group them on their
    own, so later burns inside an old fire's footprint don't extend that fire.
    Repeats until no event has late detections."""
    event = event.copy()
    t = (d.time - pd.Timestamp("2000-01-01")).dt.total_seconds().values / 86400
    for _ in range(max_rounds):
        s = pd.Series(t)
        q = s.groupby(event).quantile([0.025, 0.975]).unstack()
        lo, hi = q[0.025].reindex(event).values, q[0.975].reindex(event).values
        late = (t < lo - LATE_DAYS) | (t > hi + LATE_DAYS)
        if not late.any():
            break
        sub = link(d[late], dist_km, gap_days)
        event[late] = event.max() + 1 + sub
    return event


def fire_shares(d: pd.DataFrame, event: np.ndarray) -> pd.DataFrame:
    """Per DEECA bushfire: detections, events touched, share in its largest event.

    Pieces (fire x event) more than LATE_DAYS from the main event's time span
    are later burns inside the outline, not a split of the fire, so they're
    left out of the count."""
    m = d.deeca_firekey.notna().values
    t = pd.DataFrame({"fk": d.deeca_firekey.values[m], "ev": event[m],
                      "season": d.season.values[m], "time": d.time.values[m]})
    pieces = t.groupby(["fk", "ev"]).agg(n=("time", "size"), t0=("time", "min"),
                                         t1=("time", "max")).reset_index()
    main = pieces.loc[pieces.groupby("fk").n.idxmax(), ["fk", "ev", "t0", "t1"]]
    pieces = pieces.merge(main, on="fk", suffixes=("", "_main"))
    gap = np.maximum((pieces.t0_main - pieces.t1).dt.days, (pieces.t0 - pieces.t1_main).dt.days)
    pieces = pieces[(pieces.ev == pieces.ev_main) | (gap <= LATE_DAYS)]
    per = pieces.groupby("fk").n.agg(n="sum", events="size", top="max")
    per["share"] = per.top / per.n
    per["season"] = t.groupby("fk").season.first()
    return per[per.n >= MIN_DETECTIONS_FOR_TUNING]


def score(d: pd.DataFrame, event: np.ndarray) -> dict:
    """How the events line up with DEECA bushfires."""
    per = fire_shares(d, event)
    m = d.deeca_firekey.notna().values
    t = pd.DataFrame({"fk": d.deeca_firekey.values[m], "ev": event[m]})
    t = t[t.fk.isin(per.index)]  # merges counted on all of a fire's detections
    fires_per_event = t.groupby("ev").fk.nunique()
    return {
        "fires": len(per),
        "below_95": int((per.share < SHARE_TARGET).sum()),
        "min_share": round(float(per.share.min()), 3),
        "split_any": int((per.events > 1).sum()),
        "merged_fires": int(t[t.ev.isin(fires_per_event.index[fires_per_event > 1])].fk.nunique()),
        "events_total": len(np.unique(event)),
    }


def tune(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dist in [1.5, 2, 3, 5, 7.5, 10, 15]:
        for gap in [1, 2, 3, 5, 7, 10, 14]:
            s = score(d, link(d, dist, gap))
            rows.append({"dist_km": dist, "gap_days": gap, **s})
            print(f"  {dist:>4} km {gap:>2} d: {s}", flush=True)
    return pd.DataFrame(rows)


def absorb_fragments(d: pd.DataFrame, fday: np.ndarray) -> pd.Series:
    """Merge fragment events into their host (in place on d.event); returns the
    original event id for absorbed detections (-1 otherwise).

    Fragment: a modelling event with < FRAGMENT_MAX_DAYS active fire days or
    < FRAGMENT_MAX_DETECTIONS detections. Host: a non-fragment modelling event
    of the same season with the same majority DEECA fire key, holding
    detections in the MCD64A1 patch that has most of the fragment's detections
    (at least half of them), and with all of the fragment's detections within
    LATE_DAYS of the host's main activity (the split_late rule: later burns
    inside an old fire's outline stay separate). Same season, so never across
    a fold boundary."""
    from scipy import ndimage
    from vigil.burn_history import grid
    from vigil.labels import TOUCH_CELLS, mcd_season, patches
    transform, h, w = grid()
    col, row = ~transform * (d.x.values, d.y.values)
    row, col = np.floor(row).astype(int), np.floor(col).astype(int)
    inside = (row >= 0) & (row < h) & (col >= 0) & (col < w)
    m = d.exclude_reason.isna().values
    g = d[m].groupby("event")
    n = g.size()
    days = pd.Series(fday[m]).groupby(d.event.values[m]).nunique()
    season = g.season.first()
    key = d[m].dropna(subset=["deeca_firekey"]).groupby("event").deeca_firekey.agg(
        lambda s: s.value_counts().index[0])
    frag = (days < FRAGMENT_MAX_DAYS) | (n < FRAGMENT_MAX_DETECTIONS)
    t = pd.Series((d.time - pd.Timestamp("2000-01-01")).dt.total_seconds().values[m] / 86400)
    span = t.groupby(d.event.values[m]).quantile([0.025, 0.975]).unstack()
    t_range = t.groupby(d.event.values[m]).agg(["min", "max"])
    fold_of = {s_: f for f, ss in CV_FOLDS.items() for s_ in ss} | {s_: "test" for s_ in TEST_SEASONS}
    mapping, stats, patch_of = {}, [], {}
    for (s_, k), evs in key.groupby([season.reindex(key.index), key]):
        evs = evs.index
        frags, hosts = evs[frag[evs].values], evs[~frag[evs].values]
        if not len(frags) or not len(hosts):
            continue
        if s_ not in patch_of:        # one season at a time (groups are sorted by season)
            p = patches(mcd_season(int(s_)) > 0)
            near = ndimage.grey_dilation(p, size=(2 * TOUCH_CELLS + 1,) * 2)
            patch_of = {s_: np.where(p > 0, p, near)}
        pm = patch_of[s_]
        pid = np.zeros(len(d), np.int64)
        sel = np.isin(d.event.values, evs) & inside
        pid[sel] = pm[row[sel], col[sel]]
        for f_ in frags:
            fp = pid[d.event.values == f_]
            fp = fp[fp > 0]
            if not len(fp):
                continue
            main_patch = np.bincount(fp).argmax()
            if (fp == main_patch).sum() < 0.5 * (d.event.values == f_).sum():
                continue
            in_patch = {hh: int(((d.event.values == hh) & (pid == main_patch)).sum()) for hh in hosts
                        if t_range.loc[f_, "min"] >= span.loc[hh, 0.025] - LATE_DAYS
                        and t_range.loc[f_, "max"] <= span.loc[hh, 0.975] + LATE_DAYS}
            if not in_patch:
                continue
            host = max(in_patch, key=in_patch.get)
            if in_patch[host] == 0:
                continue
            assert fold_of[s_] == fold_of[season[host]]
            mapping[f_] = host
            stats.append((s_, k, f_, host, n[f_], days[f_]))
    orig = d.event.copy()
    d["event"] = d.event.replace(mapping)
    st = pd.DataFrame(stats, columns=["season", "key", "fragment", "host", "detections", "days"])
    st.to_csv(PROCESSED / "absorbed_fragments.csv", index=False)
    print(f"Fragments absorbed: {len(st)} events ({int(st.detections.sum()) if len(st) else 0} detections) "
          f"into {st.host.nunique() if len(st) else 0} hosts; by season {st.groupby('season').size().to_dict()}")
    return orig.where(orig != d.event, -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare", help="filter + DEECA-tag all detections (cached)")
    sub.add_parser("filtercheck", help="removed detections inside DEECA bushfires")
    sub.add_parser("tune", help="threshold grid against DEECA bushfires")
    b = sub.add_parser("build")
    b.add_argument("--dist", type=float, required=True, help="link distance, km")
    b.add_argument("--gap", type=float, required=True, help="link time gap, days")
    args = parser.parse_args()

    if args.cmd == "prepare":
        prepare()
        return
    if args.cmd == "filtercheck":
        filter_check(pd.read_parquet(CACHE))
        return

    d = load_detections()
    if args.cmd == "tune":
        t = tune(d)
        t.to_csv(PROCESSED / "event_tuning.csv", index=False)
        print(t.to_string(index=False))
        return

    first = link(d, args.dist, args.gap)
    d["event"] = split_late(d, first, args.dist, args.gap)
    moved = (pd.Series(d.event.values) != pd.Series(first)).sum()
    print(f"Late detections moved to their own events: {moved:,}")
    # Renumber events 0..N-1 in time order.
    order = d.groupby("event").time.min().sort_values().index
    d["event"] = d.event.map(pd.Series(np.arange(len(order)), index=order))
    # Tag each event by where its detections fall.
    frac = d.groupby("event").deeca_type.agg(
        bushfire=lambda s: s.eq("Bushfire").mean(), burn=lambda s: s.eq("Burn").mean())
    d["event_source"] = np.select(
        [frac.bushfire.reindex(d.event).values >= DEECA_MIN_SHARE,
         frac.burn.reindex(d.event).values > 0],
        ["deeca_bushfire", "deeca_burn"], "unmatched")
    # Split by the season of each event's first detection.
    ev_season = d.groupby("event").season.min()
    ev_split = ev_season.map(lambda x: "test" if x in TEST_SEASONS else
                             "val" if x in VAL_SEASONS else "train")
    d["split"] = d.event.map(ev_split)

    # Which events are used for modelling. Same rules for every split.
    g = d.groupby("event")
    seasons = g.season.agg(["min", "max"])
    fold_of = {s_: f for f, ss in CV_FOLDS.items() for s_ in ss} | {s_: "test" for s_ in TEST_SEASONS}
    # Spanning seasons in different CV folds (or into/out of test) would leak.
    crosses = seasons["min"].map(fold_of) != seasons["max"].map(fold_of)
    active_days = g.time.apply(lambda t: t.dt.date.nunique())
    n = g.size()
    src = g.event_source.first()
    reason = pd.Series(None, index=n.index, dtype=object)
    # An event spanning into or out of val/test would leak either way it's assigned.
    reason[crosses] = "crosses_split"
    # Planned burns are lit in mild weather and kept under control.
    reason[reason.isna() & (src == "deeca_burn")] = "planned_burn"
    # Small or one-day unmatched events: mostly farm burns, and no next-day pair.
    reason[reason.isna() & (src == "unmatched") &
           ((active_days < MIN_ACTIVE_DAYS) | (n < MIN_UNMATCHED_DETECTIONS))] = "small_unmatched"
    # Unmatched events mostly on cropland: autumn stubble burns (peak Mar-May).
    # DEECA bushfires on cropland are kept, so real NSW/SA crop fires are lost
    # while Victorian ones stay (see CLAUDE.md).
    with rasterio.open(RAW / "fuel" / "bfc_fuel_types_crop.tif") as f:
        fuel = np.array([v[0] for v in f.sample(zip(d.longitude, d.latitude))])
    crop_share = pd.Series(fuel == CROP_CLASS).groupby(d.event.values).mean()
    reason[reason.isna() & (src == "unmatched") & (crop_share > MAX_CROP_SHARE)] = "cropland_unmatched"
    # Stop-start burn-off chains (unmatched only).
    fday = (fire_day(d.time) - pd.Timestamp("2000-01-01")).dt.days.values

    def bursts(days):
        u, cnt = np.unique(days, return_counts=True)
        b = np.concatenate([[0], np.cumsum(np.diff(u) > BURST_GAP)])
        sizes = np.bincount(b, weights=cnt)
        return len(sizes), sizes.max() / sizes.sum()

    bst = pd.DataFrame([(e, *bursts(fday[idx])) for e, idx in g.indices.items()],
                       columns=["event", "n_bursts", "top_burst"]).set_index("event")
    reason[reason.isna() & (src == "unmatched") & (bst.n_bursts >= BURST_MIN)
           & (bst.top_burst < BURST_MAX_TOP)] = "bursty_unmatched"
    # Documented manual exclusions.
    first_day = pd.Series(fday).groupby(d.event.values).min()
    cent = d.groupby("event")[["latitude", "longitude"]].mean()
    for m in MANUAL_EXCLUDE:
        fd0 = (pd.Timestamp(m["first_day"]) - pd.Timestamp("2000-01-01")).days
        hit = cent.index[(seasons["min"] == m["season"]) & (first_day == fd0)
                         & ((cent.latitude - m["lat"]).abs() < 0.05)
                         & ((cent.longitude - m["lon"]).abs() < 0.05)]
        if len(hit) != 1:
            raise RuntimeError(f"manual exclusion matched {len(hit)} events: {m}")
        reason[hit] = "manual"
    # Events with a similar profile, for review only (not dropped).
    grass = pd.Series(np.isin(fuel, GRASS_CLASSES)).groupby(d.event.values).mean()
    span = g.time.agg(lambda t: (t.max() - t.min()).days)
    review = reason.isna() & (span > 30) & (grass > 0.5) & (frac.bushfire < DEECA_MIN_SHARE)
    pd.DataFrame({"season": seasons["min"], "detections": n, "span_days": span,
                  "grass_share": grass, "deeca_share": frac.bushfire, "source": src,
                  "n_bursts": bst.n_bursts, "top_burst": bst.top_burst,
                  "lat": cent.latitude, "lon": cent.longitude})[review].sort_values(
        "detections", ascending=False).to_csv(PROCESSED / "review_similar_events.csv")
    print(f"Similar-profile events for review: {int(review.sum())} -> review_similar_events.csv")
    d["exclude_reason"] = d.event.map(reason)
    print("Excluded events:", reason.value_counts().to_dict())
    d["absorbed_from"] = absorb_fragments(d, fday)
    out = PROCESSED / "fire_events.parquet"
    tmp = out.with_suffix(".tmp")
    cols = ["latitude", "longitude", "x", "y", "time", "season", "frp", "daynight",
            "confidence", "deeca_type", "deeca_firekey", "event", "event_source", "split",
            "exclude_reason", "absorbed_from"]
    d[cols].to_parquet(tmp, index=False)
    tmp.rename(out)
    print(f"{d.event.nunique():,} events -> {out}")
    print(d.drop_duplicates("event").event_source.value_counts().to_string())


if __name__ == "__main__":
    main()
