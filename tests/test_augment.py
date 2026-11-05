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
