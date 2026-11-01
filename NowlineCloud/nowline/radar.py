"""Radar sources → 8-bit 'dBZ' images on the app's 2048 px web-mercator grid.

Pixel value = (dBZ + 32) * 2, 0 = no echo — exactly what the app's renderer expects.
"""
import io
import re
import datetime as dt
import numpy as np
import h5py
import requests
from PIL import Image
from . import grid

MO_BUCKET = "https://met-office-radar-obs-data.s3.eu-west-2.amazonaws.com"
OPERA_HOST = "https://s3.waw3-1.cloudferro.com"
OPERA_BUCKET = "openradar-24h"
S = requests.Session()
S.headers["User-Agent"] = "Nowline-data/1.0 (github actions)"


def _keys(xml):
    return re.findall(r"<Key>([^<]+)</Key>", xml)


# --- Listing ------------------------------------------------------------------------------

def metoffice_list(start, end):
    """[(time, key)] for Met Office UK composites between two UTC datetimes."""
    out, day = [], start.replace(hour=0, minute=0, second=0, microsecond=0)
    while day <= end:
        r = S.get(MO_BUCKET, params={"list-type": "2", "prefix": day.strftime("radar/%Y/%m/%d/"), "max-keys": "300"}, timeout=30)
        r.raise_for_status()
        for k in _keys(r.text):
            if not k.endswith("_UK.h5"):
                continue
            t = dt.datetime.strptime(k.rsplit("/", 1)[-1][:12], "%Y%m%d%H%M").replace(tzinfo=dt.timezone.utc)
            if start <= t <= end:
                out.append((t, k))
        day += dt.timedelta(days=1)
    return sorted(out)


def opera_list(start, end):
    """[(time, key)] for OPERA DBZH composites (5-min cadence)."""
    out, day = [], start.replace(hour=0, minute=0, second=0, microsecond=0)
    while day <= end:
        prefix = day.strftime("%Y/%m/%d/OPERA/COMP/")
        marker = prefix + "OPERA@" + max(start, day).strftime("%Y%m%dT%H%M")
        while True:
            r = S.get(f"{OPERA_HOST}/{OPERA_BUCKET}", params={"prefix": prefix, "marker": marker, "max-keys": "1000"}, timeout=30)
            r.raise_for_status()
            keys = _keys(r.text)
            for k in keys:
                m = re.search(r"OPERA@(\d{8}T\d{4})@0@DBZH\.h5$", k)
                if m:
                    t = dt.datetime.strptime(m.group(1), "%Y%m%dT%H%M").replace(tzinfo=dt.timezone.utc)
                    if start <= t <= end:
                        out.append((t, k))
            if len(keys) < 1000 or "<IsTruncated>true" not in r.text:
                break
            marker = keys[-1]
        day += dt.timedelta(days=1)
    return sorted(out)


# --- Decoding ------------------------------------------------------------------------------

def _dataset(h5, want_quantity):
    """Find the ODIM data array for a quantity (falls back to dataset1/data1)."""
    found = None

    def visit(name, obj):
        nonlocal found
        if found is None and isinstance(obj, h5py.Dataset) and name.endswith("/data"):
            what = obj.parent.get("what")
            q = what.attrs.get("quantity") if what is not None else None
            q = q.decode() if isinstance(q, bytes) else q
            if q == want_quantity or want_quantity is None:
                found = obj
    h5.visititems(visit)
    if found is None:
        found = h5["dataset1/data1/data"]
    what = found.parent.get("what")
    attrs = dict(what.attrs) if what is not None else {}
    return found, attrs


def _scaled(ds, attrs):
    raw = ds[()].astype(np.float64)
    gain, offset = float(attrs.get("gain", 1.0)), float(attrs.get("offset", 0.0))
    nodata, undetect = attrs.get("nodata"), attrs.get("undetect")
    bad = np.zeros(raw.shape, bool)
    if nodata is not None:
        bad |= raw == float(nodata)
    if undetect is not None:
        bad |= raw == float(undetect)
    vals = raw * gain + offset
    vals[bad] = np.nan
    return vals


def encode(dbz_source, grid_name):
    """Reproject a source dBZ grid (rows × cols, NaN = none) onto the app grid → uint8."""
    idx = grid.lookup(grid_name)
    flat = dbz_source.ravel()
    out = np.zeros(idx.shape, np.uint8)
    ok = idx >= 0
    v = np.full(idx.shape, np.nan)
    v[ok] = flat[idx[ok]]
    good = np.isfinite(v)
    out[good] = np.clip((v[good] + 32) * 2, 1, 255).astype(np.uint8)
    return out


def metoffice_frame(key):
    r = S.get(f"{MO_BUCKET}/{key}", timeout=60)
    r.raise_for_status()
    with h5py.File(io.BytesIO(r.content), "r") as h5:
        ds, attrs = _dataset(h5, None)
        rate = _scaled(ds, attrs)
    rate[(rate <= 0.01) | (rate < 0)] = np.nan
    dbz = 10 * np.log10(200 * np.power(rate, 1.6))      # Marshall–Palmer
    return encode(dbz, "metoffice")


def opera_frame(key):
    r = S.get(f"{OPERA_HOST}/{OPERA_BUCKET}/{key}", timeout=90)
    r.raise_for_status()
    with h5py.File(io.BytesIO(r.content), "r") as h5:
        ds, attrs = _dataset(h5, "DBZH")
        dbz = _scaled(ds, attrs)
    dbz[(dbz < 4) | (dbz >= 90) | (dbz <= -1e5)] = np.nan   # drop clutter / sentinels
    return encode(dbz, "opera")


def save_png(arr, path):
    Image.fromarray(arr, mode="L").save(path, optimize=True)
