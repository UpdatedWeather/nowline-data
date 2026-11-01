"""The app's map domain and the source radar grids, with the same maths the iOS app uses."""
import math
import numpy as np

# UK & Ireland at web-mercator zoom 6, tiles x 29…32, y 18…21, 512 px each → 2048 × 2048.
ZOOM, X0, X1, Y0, Y1, TILE = 6, 29, 32, 18, 21, 512
SIZE = (X1 - X0 + 1) * TILE
MERC_MIN = (X0 / 2 ** ZOOM, Y0 / 2 ** ZOOM)
MERC_MAX = ((X1 + 1) / 2 ** ZOOM, (Y1 + 1) / 2 ** ZOOM)
CIRC = 40_075_016.685_578_5


def bbox_3857():
    """minx, miny, maxx, maxy in EPSG:3857 metres for WMS requests."""
    minx = (MERC_MIN[0] - 0.5) * CIRC
    maxx = (MERC_MAX[0] - 0.5) * CIRC
    maxy = (0.5 - MERC_MIN[1]) * CIRC
    miny = (0.5 - MERC_MAX[1]) * CIRC
    return minx, miny, maxx, maxy


def pixel_lonlat(size=SIZE):
    """Longitude/latitude (degrees) at the centre of every output pixel, shape (size, size)."""
    xs = MERC_MIN[0] + (MERC_MAX[0] - MERC_MIN[0]) * (np.arange(size) + 0.5) / size
    ys = MERC_MIN[1] + (MERC_MAX[1] - MERC_MIN[1]) * (np.arange(size) + 0.5) / size
    lon = xs * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(math.pi * (1 - 2 * ys))))
    return np.meshgrid(lon, lat)


def merc_to_lonlat(mx, my):
    return mx * 360.0 - 180.0, np.degrees(np.arctan(np.sinh(math.pi * (1 - 2 * my))))


# --- Projections (vectorised ports of the Swift code) -------------------------------------

def osgb(lat, lon):
    a, b, f0 = 6_377_563.396, 6_356_256.909, 0.999601
    lat0, lon0, e0, n0 = math.radians(49.0), math.radians(-2.0), 400_000.0, -100_000.0
    e2 = (a * a - b * b) / (a * a)
    n = (a - b) / (a + b)
    p, l = np.radians(lat), np.radians(lon)
    s, c, t = np.sin(p), np.cos(p), np.tan(p)
    nu = a * f0 / np.sqrt(1 - e2 * s * s)
    rho = a * f0 * (1 - e2) / (1 - e2 * s * s) ** 1.5
    eta2 = nu / rho - 1
    dp, sp = p - lat0, p + lat0
    n2, n3 = n * n, n ** 3
    m = (1 + n + 1.25 * n2 + 1.25 * n3) * dp
    m -= (3 * n + 3 * n2 + 21 / 8 * n3) * np.sin(dp) * np.cos(sp)
    m += (15 / 8 * n2 + 15 / 8 * n3) * np.sin(2 * dp) * np.cos(2 * sp)
    m -= 35 / 24 * n3 * np.sin(3 * dp) * np.cos(3 * sp)
    m *= b * f0
    c3, c5, t2 = c ** 3, c ** 5, t * t
    t4 = t2 * t2
    i1 = m + n0
    i2 = nu / 2 * s * c
    i3 = nu / 24 * s * c3 * (5 - t2 + 9 * eta2)
    i3a = nu / 720 * s * c5 * (61 - 58 * t2 + t4)
    i4 = nu * c
    i5 = nu / 6 * c3 * (nu / rho - t2)
    i6 = nu / 120 * c5 * (5 - 18 * t2 + t4 + 14 * eta2 - 58 * t2 * eta2)
    d = l - lon0
    d2 = d * d
    north = i1 + i2 * d2 + i3 * d2 * d2 + i3a * d2 ** 3
    east = e0 + i4 * d + i5 * d2 * d + i6 * d2 * d2 * d
    return east, north


def laea_opera(lat, lon, lat0=55.0, lon0=10.0, x0=1_950_000.0, y0=-2_100_000.0):
    a, f = 6_378_137.0, 1 / 298.257223563
    e2 = 2 * f - f * f
    e = math.sqrt(e2)

    def q(sinp):
        return (1 - e2) * (sinp / (1 - e2 * sinp * sinp) - (1 / (2 * e)) * np.log((1 - e * sinp) / (1 + e * sinp)))

    qp = q(1.0)
    p1 = math.radians(lat0)
    beta1 = math.asin(q(math.sin(p1)) / qp)
    rq = a * math.sqrt(qp / 2)
    m1 = math.cos(p1) / math.sqrt(1 - e2 * math.sin(p1) ** 2)
    dd = a * m1 / (rq * math.cos(beta1))
    beta = np.arcsin(np.clip(q(np.sin(np.radians(lat))) / qp, -1, 1))
    dl = np.radians(lon - lon0)
    bb = rq * np.sqrt(2 / (1 + math.sin(beta1) * np.sin(beta) + math.cos(beta1) * np.cos(beta) * np.cos(dl)))
    x = x0 + bb * dd * np.cos(beta) * np.sin(dl)
    y = y0 + (bb / dd) * (math.cos(beta1) * np.sin(beta) - math.sin(beta1) * np.cos(beta) * np.cos(dl))
    return x, y


GRIDS = {
    # rows, cols, originX, originY, cell, projection
    "metoffice": (2175, 1725, -405_000.0, 1_550_000.0, 1000.0, osgb),
    "opera": (4400, 3800, 0.0, 0.0, 1000.0, laea_opera),
}

_LUT = {}


def lookup(grid):
    """Flat source index (or -1) for every output pixel."""
    if grid in _LUT:
        return _LUT[grid]
    rows, cols, ox, oy, cell, proj = GRIDS[grid]
    lon, lat = pixel_lonlat()
    x, y = proj(lat, lon)
    col = np.floor((x - ox) / cell).astype(np.int64)
    row = np.floor((oy - y) / cell).astype(np.int64)
    ok = (col >= 0) & (col < cols) & (row >= 0) & (row < rows)
    idx = np.where(ok, row * cols + col, -1)
    _LUT[grid] = idx
    return idx
