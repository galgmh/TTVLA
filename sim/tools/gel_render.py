"""GelSight-style photometric rendering of a contact-height field.

Height field (metres, 0 = out of contact) -> surface normals -> three coloured
LEDs at 120 deg, which is why a real GelSight frame looks RGB: the colour at a
pixel encodes the *surface normal*, not the albedo. Used by
``scripts/render_video.py`` and by any dataloader that wants an image view of
the recorded marker field (``sim/tools/tactile_field.py`` produces the
field).

Pillow is imported lazily inside the functions, so importing this module costs
nothing but numpy.
"""

from __future__ import annotations

import numpy as np

# Three coloured LEDs at 120 deg around the sensor axis — the illumination a real
# GelSight uses, which is why its raw frames are RGB-looking.
GEL_LIGHTS = np.asarray(
    [[-0.62, -0.50, 0.60], [0.70, -0.42, 0.58], [0.02, 0.86, 0.51]], dtype=np.float32
)
GEL_LIGHTS /= np.linalg.norm(GEL_LIGHTS, axis=1, keepdims=True)
# Each LED lights a narrow lobe (a GelSight frame is dark gel with bright
# coloured slopes, not a lambertian pastel), plus a broad term so a flat,
# untouched pad still reads as grey rather than black.
GEL_AMBIENT, GEL_DIFFUSE, GEL_SPECULAR, GEL_SPEC_POW = 0.09, 0.35, 1.8, 5.0


def smooth(a: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur with edge padding (numpy only — Pillow's own blur
    does not accept float images, and this field is in metres)."""
    if sigma <= 0:
        return a
    r = max(1, int(round(3.0 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    a = np.apply_along_axis(np.convolve, 0, np.pad(a, ((r, r), (0, 0)), mode="edge"), k, "valid")
    a = np.apply_along_axis(np.convolve, 1, np.pad(a, ((0, 0), (r, r)), mode="edge"), k, "valid")
    return a.astype(np.float32, copy=False)


def gel_field(
    height: np.ndarray,
    size: tuple[int, int],
    *,
    hmax: float | None = None,
    blur: float | None = None,
) -> tuple[np.ndarray, float]:
    """Resize + smooth a contact-height field (metres) onto the output grid.

    Returns the field normalized so 1.0 = ``hmax`` (an episode-wide ceiling, so the
    imprint is comparable frame to frame) and that ceiling in metres, which the
    slope computation needs to get back to physical units.
    """
    from PIL import Image

    w, h = int(size[0]), int(size[1])
    hf = np.asarray(height, dtype=np.float32)
    scale_m = max(float(hmax) if hmax is not None else float(np.abs(hf).max()), 1e-12)
    if blur is None:      # one probe cell wide: the gel is smooth, the grid is not
        blur = 0.7 * min(w / max(hf.shape[1], 1), h / max(hf.shape[0], 1))
    img = Image.fromarray(hf).resize((w, h), Image.BICUBIC)   # float32 -> mode "F"
    hr = np.asarray(img, dtype=np.float32) / scale_m          # 1 = deepest contact seen
    return np.clip(smooth(hr, blur), 0.0, 1.2), scale_m


def gel_slope(hr: np.ndarray, scale_m: float, m_per_px: tuple[float, float] | None):
    """Surface slope of a normalized gel field, in m/m when the pixel scale is known."""
    gy, gx = np.gradient(hr)                               # rows, cols -> per output pixel
    if m_per_px is not None:
        gx, gy = gx * scale_m / m_per_px[0], gy * scale_m / m_per_px[1]   # -> true slope (m/m)
    return gx, gy


def gelsight_image(
    height: np.ndarray,
    size: tuple[int, int],
    m_per_px: tuple[float, float] | None = None,
    *,
    hmax: float | None = None,
    gain: float = 2.0,
    blur: float | None = None,
):
    """Contact-height map (metres, 0 = out of contact) -> GelSight-style PIL image.

    ``size`` is the output (width, height) in pixels. The field is bicubic-upsampled
    and blurred before shading: gel is continuous, so a 12-taxel field reads as a
    smooth imprint rather than 12 blocks. The height gradient becomes a surface
    normal — a *physical* slope when ``m_per_px`` is given — and the normal is lit
    by the three coloured LEDs above. ``hmax`` pins the height scale across the
    episode (default: this frame's own peak).

    This *renders* the recorded field; it invents no spatial detail. A denser
    recorded field (the `elastomer` marker grid, or the height map synthesized
    from it) feeds the same function unchanged.
    """
    from PIL import Image

    hr, scale_m = gel_field(height, size, hmax=hmax, blur=blur)
    gx, gy = gel_slope(hr, scale_m, m_per_px)
    nx, ny, nz = -gx * gain, -gy * gain, np.ones_like(hr)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    normal = np.stack([nx / norm, ny / norm, nz / norm], axis=-1)   # (h, w, 3)

    dot = np.clip(normal @ GEL_LIGHTS.T, 0.0, None)       # (h, w, 3): one channel per LED
    rgb = np.clip(GEL_AMBIENT + GEL_DIFFUSE * dot + GEL_SPECULAR * dot ** GEL_SPEC_POW, 0.0, 1.0)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def grid_m_per_px(pos: np.ndarray, grid: tuple[int, int], size: tuple[int, int]):
    """Metres per output pixel for one finger's probe grid, from the stored probe
    positions (layouts/probe_local_pos). Uses the outer probe spacing plus half a
    cell of margin, which is the physical span the resized image covers."""
    if min(grid) < 2:
        return None
    p = np.asarray(pos, dtype=np.float64).reshape(*grid, 3)
    along = float(np.linalg.norm(p[-1].mean(axis=0) - p[0].mean(axis=0)))     # rows
    across = float(np.linalg.norm(p[:, -1].mean(axis=0) - p[:, 0].mean(axis=0)))  # cols
    span_x = across * grid[1] / (grid[1] - 1)
    span_y = along * grid[0] / (grid[0] - 1)
    return span_x / size[0], span_y / size[1]


def lattice_nodes(size: tuple[int, int], grid: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Image-space (x, y) of each marker's *undeformed* node, ``(ny, nx)`` grids.

    Uses the same linear span model as :func:`grid_m_per_px` — the grid covers
    the image with half a cell of margin — so a displacement converted to pixels
    by ``m_per_px`` lands where the shading says it should.
    """
    w, h = int(size[0]), int(size[1])
    ny, nx = int(grid[0]), int(grid[1])
    xs = (np.arange(nx) + 0.5) * w / nx
    ys = (np.arange(ny) + 0.5) * h / ny
    return np.meshgrid(xs, ys, indexing="xy")


def pad_tangents(pos: np.ndarray, grid: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors along the grid's columns and rows, in the link frame.

    Derived from the stored probe positions (they follow the pad surface, so the
    mean edge-to-edge direction is the tangent), which is what maps an in-plane
    marker displacement onto the rendered image axes.
    """
    p = np.asarray(pos, dtype=np.float64).reshape(*grid, 3)
    u_col = p[:, -1].mean(axis=0) - p[:, 0].mean(axis=0)
    u_row = p[-1].mean(axis=0) - p[0].mean(axis=0)
    u_col /= max(float(np.linalg.norm(u_col)), 1e-12)
    u_row /= max(float(np.linalg.norm(u_row)), 1e-12)
    return u_col, u_row
