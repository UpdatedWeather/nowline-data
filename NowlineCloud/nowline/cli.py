"""Build the files the Consensus app downloads.

    python -m nowline.cli live   --out public --cache cache     # last 3 h, run every ~10 min
    python -m nowline.cli events --out public --cache cache     # curated history (once per event)

Everything is plain static files, published to GitHub Pages:
    latest.json                       live manifest (radar, satellite, lightning)
    live/r_<unix>.png                 radar frame, 2048² grey PNG, value = (dBZ+32)*2
    live/s_<unix>.jpg                 GeoColour satellite, 1536² JPEG
    live/lightning.json               strikes for the last 3 h
    events/index.json                 history feed (cards, facts, thumbnails)
    events/<id>/manifest.json         one event's frames
"""
import argparse
import datetime as dt
import json
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
from . import grid, radar, imagery, events as ev_mod

UTC = dt.timezone.utc
VERSION = 1


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ts(t):
    return int(t.timestamp())


def pmap(fn, items, workers=8):
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(fn, items))


# Time budget for history work in this run, and whether anything was skipped or failed
# (so an event is only marked finished once everything it can have has been fetched).
DEADLINE = [None]
INCOMPLETE = [False]


def cached(cache, name, build):
    """Return cached bytes for `name`, building (and caching) them if needed. None = unavailable."""
    path = os.path.join(cache, name)
    if os.path.exists(path):
        return path if os.path.getsize(path) > 0 else None
    if build is None:
        return None
    if DEADLINE[0] is not None and time.time() > DEADLINE[0]:
        INCOMPLETE[0] = True          # out of time: fetch it next run
        return None
    try:
        data = build()
    except Exception as e:
        log("  failed", name, e)
        INCOMPLETE[0] = True          # transient error: retry next run
        return None
    if data is None:
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return path


def png_bytes(arr):
    import io
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, "PNG", optimize=True)
    return buf.getvalue()


# --- Live ----------------------------------------------------------------------------------

def build_live(out, cache, hours=3.0):
    now = dt.datetime.now(UTC)
    start = now - dt.timedelta(hours=hours, minutes=10)
    os.makedirs(os.path.join(out, "live"), exist_ok=True)

    # Radar: OPERA every 10 minutes (the app's live cadence).
    frames = [(t, k) for t, k in radar.opera_list(start, now) if t.minute % 10 == 0]
    log(f"radar: {len(frames)} frames")

    def do_radar(item):
        t, k = item
        p = cached(cache, f"opera/r_{ts(t)}.png", lambda: png_bytes(radar.opera_frame(k)))
        return (ts(t), p)
    radar_done = [r for r in pmap(do_radar, frames, 4) if r[1]]

    # Satellite: every 10 minutes, newest first (latest slot appears ~20 min late).
    slots = sorted({ts(t) // 600 * 600 for t, _ in frames} | {ts(now) // 600 * 600 - k * 600 for k in range(1, 5)})
    slots = [s for s in slots if s <= ts(now) - 600]

    def do_sat(s):
        when = dt.datetime.fromtimestamp(s, UTC)
        p = cached(cache, f"sat/s_{s}.jpg", lambda: imagery.satellite(when))
        return (s, p)
    sat_done = [r for r in pmap(do_sat, slots, 6) if r[1]]
    log(f"satellite: {len(sat_done)}/{len(slots)}")

    # Lightning: 5-minute slots → individual strikes.
    li_slots = list(range((ts(start) // 300) * 300, ts(now) - 300, 300))

    def do_li(s):
        p = cached(cache, f"li/l_{s}.json", lambda: (lambda v: None if v is None else json.dumps(v).encode())(
            imagery.lightning(dt.datetime.fromtimestamp(s, UTC))))
        return (s, p)
    li_done = [r for r in pmap(do_li, li_slots, 8) if r[1]]
    strikes = []
    for _, p in li_done:
        with open(p) as f:
            strikes += json.load(f)
    log(f"lightning: {len(li_done)}/{len(li_slots)} slots, {len(strikes)} strikes")

    # Publish.
    manifest = {"version": VERSION, "generated": ts(now), "source": "EUMETNET OPERA",
                "domain": {"zoom": grid.ZOOM, "x": [grid.X0, grid.X1], "y": [grid.Y0, grid.Y1], "size": grid.SIZE},
                "radar": [], "satellite": [], "lightning": None,
                "lightningLatest": max((s for s, _ in li_done), default=None)}
    for t, p in radar_done:
        name = f"live/r_{t}.png"
        shutil.copy(p, os.path.join(out, name))
        manifest["radar"].append({"t": t, "url": name})
    for s, p in sat_done:
        name = f"live/s_{s}.jpg"
        shutil.copy(p, os.path.join(out, name))
        manifest["satellite"].append({"t": s, "url": name})
    with open(os.path.join(out, "live/lightning.json"), "w") as f:
        json.dump(strikes, f, separators=(",", ":"))
    manifest["lightning"] = "live/lightning.json"
    manifest["nowcast"] = build_nowcast(out, radar_done)
    with open(os.path.join(out, "latest.json"), "w") as f:
        json.dump(manifest, f, separators=(",", ":"))
    prune(cache, keep_after=ts(now) - 6 * 3600)


def build_nowcast(out, radar_done):
    """Next two hours from the newest radar frames (see nowcast.py). Never breaks the live feed."""
    try:
        from . import nowcast
    except Exception as e:
        log("nowcast: unavailable", e)
        return None
    try:
        done = sorted(radar_done)
        if len(done) < 2:
            return None
        # The newest run of frames exactly 10 minutes apart (up to 4).
        chain = [done[-1]]
        for t, p in reversed(done[:-1]):
            if chain[0][0] - t != 600 or len(chain) == 4:
                break
            chain.insert(0, (t, p))
        if len(chain) < 2:
            log("nowcast: no consecutive frames")
            return None
        frames = [np.asarray(Image.open(p).convert("L")) for _, p in chain]
        t0 = time.time()
        fc, info = nowcast.forecast(frames)
        if fc is None:
            log("nowcast: too little rain to track")
            return {"base": chain[-1][0], "step": 600, "frames": [], "dry": True}
        base = chain[-1][0]
        items = []
        for k, arr in enumerate(fc, start=1):
            t = base + k * 600
            name = f"live/n_{base}_{t}.png"          # unique per run, so no cache serves an old one
            with open(os.path.join(out, name), "wb") as f:
                f.write(png_bytes(arr))
            items.append({"t": t, "url": name})
        log(f"nowcast: {len(items)} frames from {len(chain)} inputs in {time.time() - t0:.1f}s, "
            f"motion {info['speed'] * 4 * 0.69 * 6:.0f} km/h")
        return {"base": base, "step": 600, "frames": items, "motion": info}
    except Exception:
        traceback.print_exc()
        return None


def prune(cache, keep_after):
    for sub in ("opera", "sat", "li"):
        d = os.path.join(cache, sub)
        if not os.path.isdir(d):
            continue
        for n in os.listdir(d):
            try:
                t = int(n.split("_")[1].split(".")[0])
            except Exception:
                continue
            if t < keep_after:
                os.remove(os.path.join(d, n))


# --- Events --------------------------------------------------------------------------------

PALETTE = [(4, (30, 70, 200)), (15, (44, 120, 234)), (23, (34, 188, 90)), (28, (154, 217, 58)),
           (33, (246, 226, 52)), (38, (247, 162, 44)), (42, (232, 67, 42)), (47, (190, 30, 78)), (52, (224, 79, 216))]


def colourise(arr):
    dbz = arr.astype(np.float32) / 2 - 32
    rgb = np.zeros(arr.shape + (3,), np.uint8)
    a = np.zeros(arr.shape, np.uint8)
    for lim, col in PALETTE:
        m = (arr > 0) & (dbz >= lim)
        rgb[m] = col
        a[m] = 230
    return rgb, a


def thumbnail(event, radar_png, sat_jpg, out_path, size=360):
    """Event card image: peak radar over the satellite, cropped around the event."""
    zoom = 2 ** grid.ZOOM
    mx = (event["lon"] + 180) / 360
    import math
    my = (1 - math.log(math.tan(math.radians(event["lat"])) + 1 / math.cos(math.radians(event["lat"]))) / math.pi) / 2
    px = (mx - grid.MERC_MIN[0]) / (grid.MERC_MAX[0] - grid.MERC_MIN[0]) * grid.SIZE
    py = (my - grid.MERC_MIN[1]) / (grid.MERC_MAX[1] - grid.MERC_MIN[1]) * grid.SIZE
    half = 380
    box = (int(max(0, px - half)), int(max(0, py - half)), int(min(grid.SIZE, px + half)), int(min(grid.SIZE, py + half)))
    base = Image.new("RGB", (grid.SIZE, grid.SIZE), (8, 12, 26))
    if sat_jpg:
        base = Image.open(sat_jpg).convert("RGB").resize((grid.SIZE, grid.SIZE))
        base = Image.eval(base, lambda v: int(v * 0.6))
    arr = np.asarray(Image.open(radar_png))
    rgb, a = colourise(arr)
    layer = Image.fromarray(np.dstack([rgb, a]), "RGBA")
    base = base.convert("RGBA")
    base.alpha_composite(layer)
    base.crop(box).convert("RGB").resize((size, size), Image.LANCZOS).save(out_path, "JPEG", quality=80, optimize=True)


def rebuild_event_files(e, out, cache):
    """Fast path for finished events: copy cached files into the site without any downloads."""
    try:
        return build_event(e, out, cache, offline=True)
    except Exception:
        traceback.print_exc()
        return None


def build_event(e, out, cache, offline=False):
    eid = e["id"]

    def get(name, build):
        return cached(cache, name, None if offline else build)
    start = dt.datetime.strptime(e["start"], "%Y-%m-%dT%H:%MZ").replace(tzinfo=UTC)
    end = start + dt.timedelta(hours=e["hours"])
    e = dict(e, start=ts(start))
    edir = os.path.join(out, "events", eid)
    os.makedirs(edir, exist_ok=True)
    log(f"event {eid}")

    list_path = os.path.join(cache, f"events/{eid}/frames.json")
    if offline and os.path.exists(list_path):
        frames = [(dt.datetime.fromtimestamp(t, UTC), k) for t, k in json.load(open(list_path))]
    else:
        frames = radar.metoffice_list(start, end)
        if len(frames) > 72:                              # 30-minute steps for long events
            frames = frames[::2]
        os.makedirs(os.path.dirname(list_path), exist_ok=True)
        with open(list_path, "w") as f:
            json.dump([[ts(t), k] for t, k in frames], f)

    def do_radar(item):
        t, k = item
        p = get(f"events/{eid}/r_{ts(t)}.png", lambda: png_bytes(radar.metoffice_frame(k)))
        return (ts(t), p)
    radar_done = [r for r in pmap(do_radar, frames, 6) if r[1]]

    sat_slots = sorted({ts(t) // 1800 * 1800 for t, _ in frames})

    def do_sat(s):
        p = get(f"events/{eid}/s_{s}.jpg", lambda: imagery.satellite(dt.datetime.fromtimestamp(s, UTC), size=1280))
        return (s, p)
    sat_done = [r for r in pmap(do_sat, sat_slots, 6) if r[1]]

    strikes = []
    if end > dt.datetime(2025, 5, 30, 15, tzinfo=UTC):
        li_slots = list(range(ts(start) // 300 * 300, ts(end) + 1, 300))

        def do_li(s):
            p = get(f"events/{eid}/l_{s}.json", lambda: (lambda v: None if v is None else json.dumps(v).encode())(
                imagery.lightning(dt.datetime.fromtimestamp(s, UTC))))
            return p
        for p in pmap(do_li, li_slots, 8):
            if p:
                with open(p) as f:
                    strikes += json.load(f)

    stats_path = os.path.join(cache, f"events/{eid}/stats.json")
    if not os.path.exists(stats_path):
        try:
            st = ev_mod.stats(e)
            os.makedirs(os.path.dirname(stats_path), exist_ok=True)
            with open(stats_path, "w") as f:
                json.dump(st, f)
        except Exception as ex:
            log("  stats failed", ex)
    st = json.load(open(stats_path)) if os.path.exists(stats_path) else None

    manifest = {"version": VERSION, "id": eid, "source": "Met Office", "radar": [], "satellite": [],
                "lightning": f"events/{eid}/lightning.json", "stats": st}
    for t, p in radar_done:
        shutil.copy(p, os.path.join(edir, f"r_{t}.png"))
        manifest["radar"].append({"t": t, "url": f"events/{eid}/r_{t}.png"})
    for s, p in sat_done:
        shutil.copy(p, os.path.join(edir, f"s_{s}.jpg"))
        manifest["satellite"].append({"t": s, "url": f"events/{eid}/s_{s}.jpg"})
    with open(os.path.join(edir, "lightning.json"), "w") as f:
        json.dump(strikes, f, separators=(",", ":"))
    with open(os.path.join(edir, "manifest.json"), "w") as f:
        json.dump(manifest, f, separators=(",", ":"))

    # Card thumbnail: the frame with the most rain.
    thumb = None
    if radar_done:
        best = max(radar_done, key=lambda r: int((np.asarray(Image.open(r[1])) > 2 * (25 + 32)).sum()))
        sat = min(sat_done, key=lambda s: abs(s[0] - best[0]))[1] if sat_done else None
        try:
            thumbnail(e, best[1], sat, os.path.join(edir, "thumb.jpg"))
            thumb = f"events/{eid}/thumb.jpg"
        except Exception as ex:
            log("  thumb failed", ex)
    card = {k: e[k] for k in ("id", "name", "detail", "kind", "start", "hours", "lat", "lon", "zoom")}
    card.update({"manifest": f"events/{eid}/manifest.json", "thumb": thumb, "frames": len(radar_done),
                 "strikes": len(strikes), "peaks": st["peaks"] if st else None})
    return card


def build_events(out, cache, catalogue="events.json", budget=600):
    """Builds events within a time budget; unfinished ones carry on next run (progress is cached)."""
    with open(catalogue) as f:
        evs = json.load(f)
    deadline = time.time() + budget
    cards = []
    for e in evs:
        done_marker = os.path.join(cache, f"events/{e['id']}/card.json")
        if os.path.exists(done_marker):
            card = json.load(open(done_marker))
            card = rebuild_event_files(e, out, cache) or card
            cards.append(card)
            continue
        if time.time() > deadline:
            log(f"event {e['id']}: postponed to next run")
            continue
        try:
            DEADLINE[0] = deadline
            INCOMPLETE[0] = False
            card = build_event(e, out, cache)
            DEADLINE[0] = None
            if card.get("frames"):
                cards.append(card)                    # show what we have so far
                if not INCOMPLETE[0]:
                    with open(done_marker, "w") as f:
                        json.dump(card, f)
                else:
                    log(f"event {e['id']}: partly built, finishing next run")
        except Exception:
            DEADLINE[0] = None
            traceback.print_exc()
    os.makedirs(os.path.join(out, "events"), exist_ok=True)
    with open(os.path.join(out, "events/index.json"), "w") as f:
        json.dump({"version": VERSION, "events": cards}, f, separators=(",", ":"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["live", "events", "all"])
    ap.add_argument("--out", default="public")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--budget", type=int, default=600, help="seconds to spend on new history events")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.cache, exist_ok=True)
    if a.cmd in ("live", "all"):
        build_live(a.out, a.cache)          # live first: it must never wait on history
    if a.cmd in ("events", "all"):
        build_events(a.out, a.cache, budget=a.budget)
    with open(os.path.join(a.out, "index.html"), "w") as f:
        f.write("<!doctype html><title>Consensus data</title><p>Data feed for the Consensus weather radar app.</p>")
    with open(os.path.join(a.out, ".nojekyll"), "w") as f:
        f.write("")


if __name__ == "__main__":
    sys.exit(main())
