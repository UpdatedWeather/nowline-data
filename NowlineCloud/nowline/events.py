"""Curated past events: peak numbers from Open-Meteo's archive and nearby place names."""
import json
import math
import time
import numpy as np
import requests

S = requests.Session()
S.headers["User-Agent"] = "Nowline-data/1.0 (github actions; contact via repository)"


def _grid(lat, lon, n=12, dlat=2.6, dlon=4.0):
    lats = np.linspace(min(lat + dlat, 60.9), max(lat - dlat, 49.8), n)
    lons = np.linspace(lon - dlon, lon + dlon, n)
    gl, gn = np.meshgrid(lats, lons, indexing="ij")
    return gl.ravel(), gn.ravel(), abs(lats[1] - lats[0])


def place_name(lat, lon):
    """Nearest town/village via OpenStreetMap Nominatim (1 request/second policy)."""
    try:
        time.sleep(1.1)
        r = S.get("https://nominatim.openstreetmap.org/reverse",
                  params={"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 10}, timeout=20)
        a = r.json().get("address", {})
        return a.get("town") or a.get("city") or a.get("village") or a.get("county") or a.get("state")
    except Exception:
        return None


def stats(ev):
    """Peak rain total, wettest hour, top gust, fresh snow and lying snow (with places), plus snow cover."""
    lats, lons, spacing = _grid(ev["lat"], ev["lon"])
    start = time.strftime("%Y-%m-%d", time.gmtime(ev["start"]))
    end = time.strftime("%Y-%m-%d", time.gmtime(ev["start"] + ev["hours"] * 3600))
    body = {
        "latitude": [round(float(v), 3) for v in lats], "longitude": [round(float(v), 3) for v in lons],
        "hourly": ["precipitation", "snowfall", "snow_depth", "wind_gusts_10m"],
        "start_date": [start], "end_date": [end], "wind_speed_unit": "mph",
        "timeformat": "unixtime", "timezone": ["GMT"], "cell_selection": "land",
    }
    for attempt in range(4):
        r = S.post("https://historical-forecast-api.open-meteo.com/v1/forecast", json=body, timeout=90)
        if r.status_code == 200:
            break
        time.sleep(30 * (attempt + 1))
    r.raise_for_status()
    locs = r.json()
    t0, t1 = ev["start"], ev["start"] + ev["hours"] * 3600
    peaks = {k: {"value": 0.0} for k in ("rainTotal", "peakRate", "snowfall", "snowDepth", "gust")}
    cover = []
    for i, loc in enumerate(locs):
        h = loc["hourly"]
        idx = [j for j, t in enumerate(h["time"]) if t0 <= t <= t1]
        def series(k): return [h[k][j] or 0 for j in idx]
        p, s, d, g = series("precipitation"), series("snowfall"), series("snow_depth"), series("wind_gusts_10m")
        vals = {"rainTotal": sum(p), "peakRate": max(p, default=0), "snowfall": sum(s),
                "snowDepth": max(d, default=0) * 100, "gust": max(g, default=0)}
        for k, v in vals.items():
            if v > peaks[k]["value"]:
                peaks[k] = {"value": round(v, 1), "lat": float(lats[i]), "lon": float(lons[i])}
        if vals["snowDepth"] >= 1:
            cover.append({"lat": float(lats[i]), "lon": float(lons[i]),
                          "times": [h["time"][j] for j in idx], "depthCm": [round((x or 0) * 100, 1) for x in d]})
    for k, pk in peaks.items():
        if pk["value"] > 0 and "lat" in pk:
            pk["place"] = place_name(pk["lat"], pk["lon"])
    return {"peaks": peaks, "snowCover": cover, "gridSpacingDeg": spacing}
