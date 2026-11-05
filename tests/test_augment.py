"""Augmentation must keep vector channels consistent with the rotated image."""

import numpy as np

from vigil.dataset import CHANNELS, rotate


def plane(gx: float, gy: float, n: int = 16) -> np.ndarray:
    """Elevation of a plane rising gx per cell east and gy per cell north
    (row 0 = north)."""
    r, c = np.mgrid[0:n, 0:n]
    return gx * c - gy * r


def gradient(z: np.ndarray) -> tuple[float, float]:
    d_row, d_col = np.gradient(z)
    return float(d_col.mean()), float(-d_row.mean())


def test_vectors_follow_rotation_and_flip():
    for gx, gy in [(1.0, 0.0), (0.3, -0.8)]:
        X = np.zeros((len(CHANNELS), 16, 16), np.float32)
        X[CHANNELS.index("elevation")] = plane(gx, gy)
        X[CHANNELS.index("grad_x")] = gx
        X[CHANNELS.index("grad_y")] = gy
        for k in range(4):
            for flip in (False, True):
                Xr, _, _ = rotate(X, X[0], X[0], k, flip)
                want = gradient(Xr[CHANNELS.index("elevation")])
                got = (Xr[CHANNELS.index("grad_x")].mean(), Xr[CHANNELS.index("grad_y")].mean())
                assert np.allclose(want, got, atol=1e-5), (gx, gy, k, flip, want, got)


def test_wind_alignment():
    from vigil.dataset import align_wind, rotate_fields, wind_angle
    n = 64
    for gx, gy, u, v in [(1.0, 0.0, 3.0, 0.0), (0.3, -0.8, -2.0, -5.0), (0.0, 1.0, 1.0, 1.0)]:
        X = np.zeros((len(CHANNELS), n, n), np.float32)
        X[CHANNELS.index("elevation")] = plane(gx, gy, n)
        X[CHANNELS.index("grad_x")], X[CHANNELS.index("grad_y")] = gx, gy
        X[CHANNELS.index("u_peak_D1")], X[CHANNELS.index("v_peak_D1")] = u, v
        phi = wind_angle(X)
        Xr = align_wind(X, phi, (32, 32))
        # wind now points up
        assert np.allclose(Xr[CHANNELS.index("u_peak_D1")], 0, atol=1e-4)
        assert np.allclose(Xr[CHANNELS.index("v_peak_D1")], np.hypot(u, v), atol=1e-4)
        # terrain gradient still matches the rotated elevation
        want = gradient(Xr[CHANNELS.index("elevation")])
        got = (Xr[CHANNELS.index("grad_x")].mean(), Xr[CHANNELS.index("grad_y")].mean())
        assert np.allclose(want, got, atol=1e-3), (gx, gy, u, v, want, got)
        # rotating back recovers the original (centre region)
        back = rotate_fields(Xr[[CHANNELS.index("elevation")]], -phi, (n, n),
                             centre_in=(15.5, 15.5), centre_out=((n - 1) / 2, (n - 1) / 2))[0]
        c = slice(n // 2 - 8, n // 2 + 8)
        assert np.allclose(back[c, c], X[CHANNELS.index("elevation")][c, c], atol=1e-3)
