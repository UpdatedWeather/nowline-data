"""EUMETSAT MTG imagery via EUMETView: GeoColour satellite and Lightning Imager flash areas."""
import io
import datetime as dt
import numpy as np
import requests
from PIL import Image
from . import grid

WMS = "https://view.eumetsat.int/geoserver/wms"
S = requests.Session()
S.headers["User-Agent"] = "Nowline-data/1.0 (github actions)"


import threading
import time

# EUMETView's archive can stall for minutes. After a run of failures, stop asking for a few
# minutes (callers retry on a later run) instead of burning the whole job on timeouts.
_lock = threading.Lock()
_fails = [0]
_paused_until = [0.0]


class Paused(Exception):
    pass


def _getmap(layer, fmt, size, when, transparent=False):
    with _lock:
        if time.time() < _paused_until[0]:
            raise Paused("EUMETView paused after repeated failures")
    try:
        content = _request(layer, fmt, size, when, transparent)
    except Exception:
        with _lock:
            _fails[0] += 1
            if _fails[0] >= 6:
                _paused_until[0] = time.time() + 240
                _fails[0] = 0
        raise
    with _lock:
        _fails[0] = 0
    return content


def _request(layer, fmt, size, when, transparent):
    minx, miny, maxx, maxy = grid.bbox_3857()
    params = {
        "service": "WMS", "version": "1.3.0", "request": "GetMap", "layers": layer, "styles": "",
        "format": fmt, "crs": "EPSG:3857", "bbox": f"{minx:.0f},{miny:.0f},{maxx:.0f},{maxy:.0f}",
        "width": size, "height": size, "time": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if transparent:
        params["transparent"] = "true"
    r = S.get(WMS, params=params, timeout=(10, 30))
    if r.status_code != 200 or "image" not in r.headers.get("Content-Type", ""):
        return None
    return r.content


def satellite(when, size=1536, quality=78):
    """GeoColour JPEG (re-encoded smaller), or None if that slot isn't published."""
    raw = _getmap("mtg_fd:rgb_geocolour", "image/jpeg", 2048, when)
    if raw is None or len(raw) < 20_000:
        return None
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    if size != 2048:
        img = img.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality, optimize=True, progressive=True)
    return buf.getvalue()


def _splitmix(seed):
    state = (seed + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    while True:
        state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        yield ((z ^ (z >> 31)) >> 11) / float(1 << 53)


def lightning(when, cell=6, window=300):
    """Individual strikes for one 5-minute slot: [[lat, lon, unixTime, heat, seed], …] or None."""
    raw = _getmap("mtg_fd:li_afa", "image/png", 2048, when, transparent=True)
    if raw is None:
        return None
    img = np.asarray(Image.open(io.BytesIO(raw)).convert("RGBA")).astype(np.float32)
    h, w = img.shape[:2]
    a = img[..., 3]
    ys, xs = np.nonzero(a > 40)
    if len(xs) == 0:
        return []
    g = img[ys, xs, 1] / 255.0
    hot = np.clip(1 - g, 0, 1)                      # pale yellow → deep red = more flashes
    cw = (w + cell - 1) // cell
    cid = (ys // cell) * cw + (xs // cell)
    order = np.argsort(cid)
    cid, xs, ys, hot = cid[order], xs[order], ys[order], hot[order]
    uniq, start, counts = np.unique(cid, return_index=True, return_counts=True)
    sx = np.add.reduceat(xs, start) / counts
    sy = np.add.reduceat(ys, start) / counts
    heat = np.maximum.reduceat(hot, start)
    t0 = int(when.timestamp())
    rng = _splitmix(t0)
    mx0, my0 = grid.MERC_MIN
    mw, mh = grid.MERC_MAX[0] - mx0, grid.MERC_MAX[1] - my0
    out = []
    for cx, cy, ht, n in zip(sx, sy, heat, counts):
        k = 1 + int(ht * 2.5) + min(int(n) // 18, 2)
        for _ in range(k):
            px = cx + (next(rng) - 0.5) * cell * 1.2
            py = cy + (next(rng) - 0.5) * cell * 1.2
            lon, lat = grid.merc_to_lonlat(mx0 + (px + 0.5) / w * mw, my0 + (py + 0.5) / h * mh)
            out.append([round(float(lat), 4), round(float(lon), 4), round(t0 - next(rng) * window, 1),
                        round(float(ht), 2), round(next(rng), 3)])
        if len(out) > 900:
            break
    return out
