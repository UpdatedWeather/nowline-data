"""Radar nowcast for the next two hours, built in the cloud so every phone shows the same thing.

The method follows the approach used by operational nowcasting systems (pysteps / rainymotion):

1. Motion. Dense optical flow (OpenCV DIS, as in rainymotion's "Dense" model) between the last
   few 10-minute frames, on a 4× reduced grid. The three frame pairs are averaged, then the field
   is smoothed with a rain-weighted Gaussian over ~50 km and relaxed towards the area-wide motion
   where there's little rain. The result varies smoothly across the UK (Scotland and the south can
   move differently) but has no small-scale wiggles, so the forecast can't tear or swirl.

2. Advection. Backward semi-Lagrangian steps, 10 minutes at a time, along that field. The
   displacement is computed on the reduced grid and applied to the full-resolution frame, so the
   rain keeps all its radar detail.

3. Predictability. Small showers are far less predictable than large rain bands (the idea behind
   S-PROG): with lead time the small scales are progressively smoothed away and slightly weakened
   while the large-scale pattern is kept, so later frames look like a forecast, not a copy.

Output frames use the same 8-bit encoding as observed radar: (dBZ + 32) × 2, 0 = no echo.
"""
import contextlib
import io
import numpy as np
import cv2
from scipy import ndimage

STEP_MIN = 10
LEADS = 12                 # 12 × 10 min = 2 hours
REDUCE = 4                 # motion grid: 2048 / 4 = 512 px (~2.8 km)
MIN_DBZ = 8.0              # echoes used for motion
SMOOTH_PX = 18             # motion smoothing on the reduced grid (~50 km)
DRY_DB = -17.0             # rain rate floor in dB (0.02 mm/h): below the weakest echo the radar shows
MIN_OUT_DBZ = 4.0          # same lower limit as the observed OPERA frames


def dbz_of(arr):
    """uint8 frame → dBZ float32, NaN where no echo."""
    a = arr.astype(np.float32)
    d = a / 2 - 32
    d[arr == 0] = np.nan
    return d


def encode(dbz):
    out = np.zeros(dbz.shape, np.uint8)
    ok = np.isfinite(dbz)
    out[ok] = np.clip((dbz[ok] + 32) * 2, 1, 255).astype(np.uint8)
    return out


def _reduced(arr):
    """Max-pool to the motion grid and scale echoes to 0…255 for optical flow."""
    h, w = arr.shape
    r = arr[: h - h % REDUCE, : w - w % REDUCE].reshape(h // REDUCE, REDUCE, w // REDUCE, REDUCE).max(axis=(1, 3))
    d = r.astype(np.float32) / 2 - 32
    v = np.where(r > 0, np.clip((d - MIN_DBZ) / (55 - MIN_DBZ), 0, 1) * 255, 0)
    # A little blur helps the flow lock onto rain areas rather than pixel noise.
    return cv2.GaussianBlur(v.astype(np.uint8), (0, 0), 1.2)


def motion(frames):
    """Smooth motion field (u, v) in reduced-grid pixels per 10 minutes, plus the area-wide mean.

    `frames`: list of uint8 frames, oldest first, 10 minutes apart (at least 2).
    Returns (u, v, (mean_u, mean_v), wet_fraction) or None if there's too little rain.
    """
    imgs = [_reduced(f) for f in frames]
    wet = imgs[-1] > 0
    if wet.mean() < 0.002:
        return None
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setUseSpatialPropagation(True)
    us, vs, ws = [], [], []
    for i in range(len(imgs) - 1):
        a, b = imgs[i], imgs[i + 1]
        flow = dis.calc(a, b, None)                     # a(x) ≈ b(x + flow)
        w = ((a > 0) | (b > 0)).astype(np.float32)
        us.append(flow[..., 0]); vs.append(flow[..., 1]); ws.append(w * (1.0 + i))   # favour recent pairs
    W = np.sum(ws, axis=0)
    U = np.sum([u * w for u, w in zip(us, ws)], axis=0) / np.maximum(W, 1e-6)
    V = np.sum([v * w for v, w in zip(vs, ws)], axis=0) / np.maximum(W, 1e-6)
    has = W > 0
    if has.sum() < 200:
        return None

    # Robust area-wide motion (median of rain pixels) and rejection of wild vectors.
    mu, mv = float(np.median(U[has])), float(np.median(V[has])),
    dev = np.hypot(U - mu, V - mv)
    spread = float(np.percentile(dev[has], 90)) + 0.5
    keep = has & (dev <= max(3.0, 2.5 * spread))
    # Flow at the edge of the grid is unreliable (rain entering or leaving): leave it out.
    m = 12
    keep[:m, :] = keep[-m:, :] = False
    keep[:, :m] = keep[:, -m:] = False
    wgt = keep.astype(np.float32) * np.minimum(W, 3.0)

    # Rain-weighted smoothing, relaxed towards the area-wide motion where data is thin.
    def smooth(x):
        return ndimage.gaussian_filter(x, SMOOTH_PX, mode="nearest")
    sw = smooth(wgt)
    prior = 0.02
    Us = (smooth(U * wgt) + prior * mu) / (sw + prior)
    Vs = (smooth(V * wgt) + prior * mv) / (sw + prior)
    # A second, gentler pass removes any remaining kinks.
    Us = ndimage.gaussian_filter(Us, SMOOTH_PX / 2, mode="nearest")
    Vs = ndimage.gaussian_filter(Vs, SMOOTH_PX / 2, mode="nearest")
    return Us.astype(np.float32), Vs.astype(np.float32), (mu, mv), float(wet.mean())


def displacements(U, V, leads=LEADS):
    """Backward displacement fields (reduced px) for each lead, by stepping 10 minutes at a time.

    The rain at x after k steps came from x - D_k(x), where D_k is traced along the field.
    """
    h, w = U.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    px, py = xx.copy(), yy.copy()
    out = []
    for _ in range(leads):
        # Velocity at the current back-traced position (bilinear), then one step back.
        u = cv2.remap(U, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        v = cv2.remap(V, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        px = px - u
        py = py - v
        out.append((xx - px, yy - py))
    return out


def _rain_db(dbz):
    """dBZ (NaN = dry) → rain rate in dB (Marshall–Palmer), dry = DRY_DB."""
    z = np.where(np.isfinite(dbz), dbz, -40.0)
    rate = np.power(np.power(10, z / 10) / 200, 1 / 1.6)
    return np.maximum(10 * np.log10(np.maximum(rate, 1e-6)), DRY_DB).astype(np.float32)


def _dbz_of_rain_db(rdb):
    rate = np.power(10, rdb / 10)
    dbz = 10 * np.log10(200 * np.power(rate, 1.6))
    dbz[dbz < MIN_OUT_DBZ] = np.nan
    return dbz


def forecast_flow(frames, leads=LEADS):
    """Fallback nowcast (OpenCV optical flow), used if pysteps isn't available or fails."""
    m = motion(frames)
    if m is None:
        return None, None
    U, V, (mu, mv), wet = m
    latest = frames[-1]
    H, W = latest.shape
    base_db = _rain_db(dbz_of(latest))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    out = []
    for k, (dx, dy) in enumerate(displacements(U, V, leads), start=1):
        # Full-resolution displacement (reduced px → full px).
        DX = cv2.resize(dx, (W, H), interpolation=cv2.INTER_LINEAR) * REDUCE
        DY = cv2.resize(dy, (W, H), interpolation=cv2.INTER_LINEAR) * REDUCE
        moved = cv2.remap(base_db, xx - DX, yy - DY, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=DRY_DB)
        lead_min = k * STEP_MIN
        # Predictability: small scales (below a size growing with lead) fade away.
        sigma = 1.0 + 0.04 * lead_min                    # full-res px (~0.7 km): ~2.5 km at 1 h, ~4 km at 2 h
        large = ndimage.gaussian_filter(moved, sigma, mode="nearest")
        keep_small = np.exp(-lead_min / 120.0)
        blended = large + (moved - large) * keep_small
        # Only rain that existed can move; smoothing must not paint drizzle over dry land.
        support = ndimage.maximum_filter(moved > DRY_DB + 0.5, size=int(2 * sigma) + 1)
        blended = np.where(support, blended, DRY_DB)
        out.append(encode(_dbz_of_rain_db(blended)))
    speed = float(np.hypot(mu, mv))
    info = {"u": mu, "v": mv, "speed": speed, "wet": wet, "method": "flow"}
    return out, info


# --- pysteps --------------------------------------------------------------------------------
#
# The main method. pysteps (Pulkkinen et al. 2019) is the open-source nowcasting library used by
# several national weather services:
#   • Lucas–Kanade optical flow on features in the rain field, interpolated to a smooth dense
#     motion field (pysteps.motion "LK");
#   • S-PROG (Seed 2003): the rain field is split into a cascade of spatial scales; each scale is
#     advected along the motion field and evolved with its own AR(2) process fitted to the last
#     three radar frames, so small, short-lived showers lose detail quickly while large rain bands
#     keep their shape; probability matching keeps realistic intensities.
# S-PROG runs on a half-resolution grid (~1.4 km). For the first frames the full-resolution radar,
# moved along the same motion field (pysteps semi-Lagrangian extrapolation), is blended in so the
# step from "now" into the forecast keeps all the radar's detail.

PS_REDUCE = 2              # S-PROG grid: 2048 / 2 = 1024 px
RAIN_THR = 0.05            # mm/h: rain / no-rain for the transforms
ZERO_DB = -15.0            # dB value for no rain
PROBMATCH = "mean"         # keep the mean rain; "cdf" flattens wide areas into one shade


def _rate(arr, reduce):
    """uint8 frame → rain rate mm/h (Marshall–Palmer), block-averaged in Z by `reduce`."""
    a = arr.astype(np.float32)
    z = np.where(arr > 0, np.power(10.0, (a / 2 - 32) / 10), 0.0)
    if reduce > 1:
        h, w = z.shape
        z = z[: h - h % reduce, : w - w % reduce].reshape(h // reduce, reduce, w // reduce, reduce).mean(axis=(1, 3))
    return np.power(z / 200.0, 1 / 1.6).astype(np.float64)


def _db(r):
    return np.where(r >= RAIN_THR, 10 * np.log10(np.maximum(r, 1e-9)), ZERO_DB)


def forecast_pysteps(frames, leads=LEADS):
    from pysteps import motion as ps_motion, nowcasts as ps_nowcasts, extrapolation as ps_extrap

    if len(frames) < 3:
        return None, None
    rates = np.stack([_rate(f, PS_REDUCE) for f in frames])
    wet = float((rates[-1] >= 0.1).mean())
    if wet < 0.002:
        return None, None
    rdb = np.stack([_db(r) for r in rates])

    # Motion: Lucas–Kanade, dense, in half-res pixels per 10 minutes.
    lk = ps_motion.get_method("LK")
    with contextlib.redirect_stdout(io.StringIO()):
        V = lk(rdb, verbose=False)
    if not np.all(np.isfinite(V)):
        V = np.nan_to_num(V)
    # Smooth over ~40 km so the field can't kink between features (kinks stretch rain into streaks),
    # and keep it near the area-wide motion at the edges, where LK has few features to go on.
    V = np.stack([ndimage.gaussian_filter(V[i], 30, mode="nearest") for i in range(2)])
    rainy = rdb[-1] > ZERO_DB
    med = [float(np.median(V[i][rainy])) if rainy.any() else 0.0 for i in range(2)]
    h, w = V.shape[1:]
    yy, xx = np.mgrid[0:h, 0:w]
    edge = np.minimum(np.minimum(xx, w - 1 - xx), np.minimum(yy, h - 1 - yy)).astype(np.float32)
    trust = np.clip(edge / 60.0, 0, 1)
    V = np.stack([V[i] * trust + med[i] * (1 - trust) for i in range(2)])

    # S-PROG on the last three frames.
    sprog = ps_nowcasts.get_method("sprog")
    with contextlib.redirect_stdout(io.StringIO()):          # pysteps prints a long report
        fc_db = sprog(rdb[-3:], V, leads, precip_thr=10 * np.log10(RAIN_THR), n_cascade_levels=6,
                  ar_order=2, probmatching_method=PROBMATCH, conditional=False)
        fc_db = np.asarray(fc_db)
    fc_db = np.where(np.isfinite(fc_db), fc_db, ZERO_DB)

    # Full-resolution extrapolation of the latest frame along the same motion (for detail early on).
    H, W = frames[-1].shape
    Vfull = np.stack([cv2.resize(V[i].astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) * PS_REDUCE
                      for i in range(2)])
    full_db = _db(_rate(frames[-1], 1))
    extrap = ps_extrap.get_method("semilagrangian")
    sharp = extrap(full_db, Vfull, leads, outval=ZERO_DB)

    out = []
    for k in range(leads):
        lead_min = (k + 1) * STEP_MIN
        coarse = cv2.resize(fc_db[k].astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        sh = np.where(np.isfinite(sharp[k]), sharp[k], ZERO_DB).astype(np.float32)
        # Early on mostly the radar itself (moved); later S-PROG's evolving large scales, with the
        # radar's fine texture laid over them (fading with lead) so rain still looks like rain.
        w = float(np.exp(-lead_min / 40.0))
        detail = (sh - cv2.GaussianBlur(sh, (0, 0), 12.0)) * float(np.exp(-lead_min / 150.0))
        mix_db = w * sh + (1 - w) * (coarse + detail * (coarse > ZERO_DB + 1))
        # Keep the rain area realistic: as much wet area as the moved radar has (S-PROG's smoothing
        # otherwise spreads a thin veil of drizzle over everything), taking the wettest pixels.
        target = int((sh > 10 * np.log10(RAIN_THR) + 0.5).sum())
        if 0 < target < mix_db.size:
            cut = np.partition(mix_db.ravel(), mix_db.size - target)[mix_db.size - target]
            mix_db = np.where(mix_db >= cut, mix_db, ZERO_DB)
        rate = np.power(10, mix_db / 10)
        dbz = 10 * np.log10(200 * np.power(np.maximum(rate, 1e-9), 1.6))
        dbz[(rate < RAIN_THR) | (dbz < MIN_OUT_DBZ)] = np.nan
        out.append(encode(dbz.astype(np.float32)))

    # Area-wide motion in the units the log expects (quarter-res px per 10 min).
    mu, mv = med[0] * PS_REDUCE / REDUCE, med[1] * PS_REDUCE / REDUCE
    info = {"u": mu, "v": mv, "speed": float(np.hypot(mu, mv)), "wet": wet, "method": "pysteps-sprog"}
    return out, info


def forecast(frames, leads=LEADS):
    """Nowcast frames (uint8, full resolution) for +10 … +leads×10 minutes.

    pysteps S-PROG when available, otherwise the optical-flow fallback above.
    Returns (list_of_frames, info) or (None, None) when there's not enough rain to track.
    """
    try:
        fc, info = forecast_pysteps(frames, leads)
        if fc is not None:
            return fc, info
        if len(frames) >= 3:
            return None, None
    except Exception as e:                                # pysteps missing or failed: fall back
        print("pysteps nowcast failed, using optical flow:", repr(e), flush=True)
    return forecast_flow(frames, leads)
