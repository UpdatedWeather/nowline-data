"""EUMETSAT MTG imagery via EUMETView: GeoColour satellite and Lightning Imager flash areas."""
import io
import datetime as dt
import numpy as np
import requests
from PIL import Image
from . import grid

WMS = "https://view.eumetsat.int/geoserver/wms"
S = requests.Session()
S.headers["User-Agent"] = "Consensus-data/1.0 (github actions)"


import threading
import time

# EUMETView's archive can stall for minutes. After a run of failures, stop asking for a few
# minutes (callers retry on a later run) instead of burning the whole job on timeouts.
_lock = threading.Lock()
_fails = [0]
_paused_until = [0.0]


class Paused(Exception):
    pass


def _getmap(layer, fmt, size, when, transparent=False, bbox=None):
    with _lock:
        if time.time() < _paused_until[0]:
            raise Paused("EUMETView paused after repeated failures")
    try:
        content = _request(layer, fmt, size, when, transparent, bbox)
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


def _request(layer, fmt, size, when, transparent, bbox=None):
    minx, miny, maxx, maxy = bbox or grid.bbox_3857()
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


# The app's wider cloud area (web-mercator zoom 6, tiles x 28…33, y 17…22), as in the app.
WIDE = (6, 28, 33, 17, 22)


def _wide_bbox():
    z, x0, x1, y0, y1 = WIDE
    n = 2 ** z
    minx = (x0 / n - 0.5) * grid.CIRC
    maxx = ((x1 + 1) / n - 0.5) * grid.CIRC
    maxy = (0.5 - y0 / n) * grid.CIRC
    miny = (0.5 - (y1 + 1) / n) * grid.CIRC
    return minx, miny, maxx, maxy


def _box_blur(a, r):
    """Two-pass box blur (edges extended), float array."""
    k = 2 * r + 1
    p = np.pad(a, ((0, 0), (r, r)), mode="edge")
    c = np.cumsum(np.pad(p, ((0, 0), (1, 0))), axis=1)
    a = (c[:, k:] - c[:, :-k]) / k
    p = np.pad(a, ((r, r), (0, 0)), mode="edge")
    c = np.cumsum(np.pad(p, ((1, 0), (0, 0))), axis=0)
    return (c[k:, :] - c[:-k, :]) / k


def cloud_cover(when, side=1024):
    """Cloudiness (0…255, PNG) over the app's wide area for a 10-minute slot, exactly as the app
    makes it: the cloud mask (finds low cloud and fog too), deepened where the infrared shows
    thick, cold tops. None if the pictures aren't out yet (or come back blank)."""
    box = _wide_bbox()
    ir_raw = _getmap("mtg_fd:ir105_hrfi", "image/jpeg", side, when, bbox=box)
    mask_time = when.replace(minute=when.minute // 15 * 15, second=0, microsecond=0)
    mask_raw = _getmap("msg_fes:clm", "image/png", side, mask_time, bbox=box)
    if mask_raw is None:
        mask_raw = _getmap("msg_fes:clm", "image/png", side, mask_time - dt.timedelta(minutes=15), bbox=box)
    ir = None
    if ir_raw is not None and len(ir_raw) > 1000:
        g = np.asarray(Image.open(io.BytesIO(ir_raw)).convert("L").resize((side, side)), np.float64) / 255
        t = np.clip((g - 0.33) / 0.42, 0, 1)
        ir = t * t * (3 - 2 * t)
    cloudy = None
    if mask_raw is not None and len(mask_raw) > 1000:
        m = np.asarray(Image.open(io.BytesIO(mask_raw)).convert("RGBA").resize((side, side), Image.NEAREST), np.float64)
        white = (m[..., 3] > 128) & (m[..., 0] > 170) & (m[..., 1] > 170) & (m[..., 2] > 170)
        cloudy = _box_blur(white.astype(np.float64), 4)
    if ir is None and cloudy is None:
        return None
    if ir is None:
        out = cloudy
    elif cloudy is None:
        out = ir
    else:
        out = np.maximum(cloudy * (0.5 + 0.5 * ir), ir)
    # A slot that isn't published yet comes back as an empty picture: not real.
    if (out > 40 / 255).mean() < 0.002:
        return None
    buf = io.BytesIO()
    Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8), mode="L").save(buf, "PNG", optimize=True)
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
