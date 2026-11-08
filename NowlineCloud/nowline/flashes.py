"""Real lightning: every flash seen by Meteosat Third Generation's Lightning Imager (LI-2-LFL,
"Lightning Flashes"), from the EUMETSAT Data Store, with its own position, time and brightness.

Needs a free EUMETSAT API key: set EUMETSAT_KEY and EUMETSAT_SECRET (GitHub → Settings → Secrets
and variables → Actions). Without them the job falls back to the flash-area picture (imagery.lightning).

Each product covers 10 minutes. Products never change once published, so each one's flashes over
the radar area are cached by product id and only new products are downloaded.
"""
import datetime as dt
import hashlib
import io
import json
import os
import zipfile
from urllib.parse import quote

import h5py
import numpy as np
import requests

from . import grid

TOKEN_URL = "https://api.eumetsat.int/token"
SEARCH_URL = "https://api.eumetsat.int/data/search-products/1.0.0/os"
DOWNLOAD_URL = "https://api.eumetsat.int/data/download/1.0.0/collections/{c}/products/{p}"
COLLECTION = "EO:EUM:DAT:0691"
EPOCH_2000 = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc).timestamp()


def credentials():
    key, secret = os.environ.get("EUMETSAT_KEY", "").strip(), os.environ.get("EUMETSAT_SECRET", "").strip()
    return (key, secret) if key and secret else None


def _token(creds):
    r = requests.post(TOKEN_URL, data={"grant_type": "client_credentials"}, auth=creds, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def _search(tok, start, end):
    """Product ids whose time span overlaps start…end."""
    ids, page = [], 0
    while True:
        params = {"pi": COLLECTION, "dtstart": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "dtend": end.strftime("%Y-%m-%dT%H:%M:%SZ"), "format": "json", "c": 100, "si": page * 100}
        r = requests.get(SEARCH_URL, params=params, headers={"Authorization": f"Bearer {tok}"}, timeout=60)
        r.raise_for_status()
        feats = r.json().get("features", [])
        ids += [f.get("id") or f.get("properties", {}).get("identifier") for f in feats]
        if len(feats) < 100:
            return [i for i in ids if i]
        page += 1


def _scaled(ds):
    """A NetCDF variable as floats, with its scale, offset and missing values applied."""
    raw = ds[()]
    a = raw.astype(np.float64)
    fill = ds.attrs.get("_FillValue")
    mask = np.zeros(a.shape, bool) if fill is None else (raw == np.asarray(fill).ravel()[0])
    a = a * float(np.asarray(ds.attrs.get("scale_factor", 1.0)).ravel()[0]) + float(np.asarray(ds.attrs.get("add_offset", 0.0)).ravel()[0])
    a[mask] = np.nan
    return a


def _flashes_in(zip_bytes):
    """[(lat, lon, unix, radiance)] for every flash over the radar area in one product."""
    lon0, lat1 = grid.merc_to_lonlat(grid.MERC_MIN[0], grid.MERC_MIN[1])
    lon1, lat0 = grid.merc_to_lonlat(grid.MERC_MAX[0], grid.MERC_MAX[1])
    out = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for name in z.namelist():
            if "BODY" not in name or not name.endswith(".nc"):
                continue
            with h5py.File(io.BytesIO(z.read(name)), "r") as f:
                if "latitude" not in f or "longitude" not in f or "flash_time" not in f:
                    continue
                lat, lon, t = _scaled(f["latitude"]), _scaled(f["longitude"]), _scaled(f["flash_time"])
                rad = _scaled(f["radiance"]) if "radiance" in f else np.ones_like(lat)
                ok = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(t) & (lat >= lat0) & (lat <= lat1) & (lon >= lon0) & (lon <= lon1)
                for a, b, c, d in zip(lat[ok], lon[ok], t[ok], rad[ok]):
                    out.append((float(a), float(b), EPOCH_2000 + float(c), float(d) if np.isfinite(d) else 1.0))
    return out


def recent(start, end, cache, log=print):
    """Every flash over the radar area from start to end, as the app expects:
    [[lat, lon, unixTime, heat 0…1, seed 0…1], …]. Returns None without credentials or on failure."""
    creds = credentials()
    if not creds:
        return None
    try:
        tok = _token(creds)
        ids = _search(tok, start, end)
    except Exception as e:
        log("flashes: search failed", e)
        return None
    folder = os.path.join(cache, "lfl")
    os.makedirs(folder, exist_ok=True)
    for name in os.listdir(folder):  # products older than the window are no longer needed
        full = os.path.join(folder, name)
        if os.path.getmtime(full) < start.timestamp() - 3600:
            os.remove(full)
    flashes, fetched = [], 0
    for pid in ids:
        path = os.path.join(cache, "lfl", hashlib.sha1(pid.encode()).hexdigest() + ".json")
        if not os.path.exists(path):
            try:
                url = DOWNLOAD_URL.format(c=quote(COLLECTION, safe=""), p=quote(pid, safe=""))
                r = requests.get(url, headers={"Authorization": f"Bearer {tok}"}, timeout=120)
                r.raise_for_status()
                items = _flashes_in(r.content)
                with open(path, "w") as fh:
                    json.dump(items, fh, separators=(",", ":"))
                fetched += 1
            except Exception as e:
                log("flashes: product failed", pid, e)
                continue
        with open(path) as fh:
            flashes += json.load(fh)
    t0, t1 = start.timestamp(), end.timestamp()
    flashes = [f for f in flashes if t0 <= f[2] <= t1]
    if not flashes:
        log(f"flashes: {len(ids)} products ({fetched} new), none over the area")
        return []
    rad = np.array([f[3] for f in flashes])
    lo, hi = np.percentile(np.log1p(rad), [5, 99]) if len(rad) > 10 else (0.0, max(np.log1p(rad).max(), 1.0))
    out = []
    for lat, lon, t, r in sorted(flashes, key=lambda f: f[2]):
        heat = float(np.clip((np.log1p(r) - lo) / max(hi - lo, 1e-6), 0, 1))
        seed = int(hashlib.md5(f"{lat:.4f}{lon:.4f}{t:.2f}".encode()).hexdigest()[:6], 16) / 0xFFFFFF
        out.append([round(lat, 4), round(lon, 4), round(t, 1), round(heat, 2), round(seed, 3)])
    log(f"flashes: {len(ids)} products ({fetched} new), {len(out)} flashes")
    return out
