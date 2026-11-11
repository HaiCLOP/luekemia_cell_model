"""Cell finder on synthetic smears with known answers: red-cell size, white-cell count, lobed cells,
fields without white cells, and fields packed with blasts."""
import numpy as np
import pytest

from src.vision.segment import CNMC_FRAME, find_white_cells, to_cnmc

RBC = (205, 160, 170)       # pale pink-lavender red cell
NUC = (95, 45, 150)         # deep purple nucleus
CYTO = (170, 165, 215)      # pale blue cytoplasm


def disk(img, cy, cx, r, colour, rng=None, texture=0):
    yy, xx = np.ogrid[: img.shape[0], : img.shape[1]]
    m = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
    val = np.array(colour, float)
    if texture and rng is not None:
        img[m] = np.clip(val + rng.normal(0, texture, (m.sum(), 3)), 0, 255)
    else:
        img[m] = val
    return m


def smear(n_wbc=3, rbc_r=14, size=(480, 640), seed=0, lobed=0, packed=False, rbc=True):
    rng = np.random.default_rng(seed)
    img = np.full((*size, 3), (245, 242, 238), float)
    if rbc:
        for _ in range(140):
            cy, cx = rng.integers(rbc_r, size[0] - rbc_r), rng.integers(rbc_r, size[1] - rbc_r)
            disk(img, cy, cx, rbc_r, RBC, rng, 4)
            disk(img, cy, cx, rbc_r * 0.4, (225, 200, 205))          # central pallor
    centres = []
    if packed:
        for y in range(70, size[0] - 60, 62):
            for x in range(70, size[1] - 60, 62):
                centres.append((y, x))
    else:
        for i in range(n_wbc):
            centres.append((120 + (i % 2) * 220, 120 + i * 170))
    for cy, cx in centres:
        disk(img, cy, cx, rbc_r * 1.5, CYTO, rng, 3)
        disk(img, cy, cx, rbc_r * 1.25, NUC, rng, 10)
    for j in range(lobed):                                           # a neutrophil: 3 small lobes
        cy, cx = 380, 140 + j * 200
        disk(img, cy, cx, rbc_r * 1.6, (215, 190, 210), rng, 3)
        for dy, dx in ((-9, -9), (-9, 10), (10, 0)):
            disk(img, cy + dy, cx + dx, rbc_r * 0.45, NUC, rng, 8)
    return np.clip(img, 0, 255).astype(np.uint8), len(centres)


def test_finds_every_white_cell_and_measures_red_cells():
    img, n = smear(n_wbc=3, rbc_r=14)
    res = find_white_cells(img)
    assert len(res.cells) == n
    assert all(c.lymphoid_like for c in res.cells)
    assert res.rbc_diameter_px == pytest.approx(28, rel=0.2)


def test_field_without_white_cells_finds_none():
    img, _ = smear(n_wbc=0)
    assert find_white_cells(img).cells == []


def test_lobed_cell_is_not_lymphoid():
    img, n = smear(n_wbc=2, lobed=1)
    res = find_white_cells(img)
    lymph = [c for c in res.cells if c.lymphoid_like]
    assert len(lymph) == n
    assert len(res.cells) == n + 1      # the three lobes come back as one cell


def test_packed_blast_field_still_works():
    img, n = smear(packed=True, rbc_r=14)
    res = find_white_cells(img, rbc_override=28)
    assert len(res.cells) >= 0.8 * n


def test_cnmc_crop_has_cnmc_geometry():
    img, _ = smear(n_wbc=1, rbc_r=14)
    res = find_white_cells(img)
    c = res.cells[0]
    crop = to_cnmc(img, c, res.scale, mode="ruler")
    assert crop.shape == (CNMC_FRAME, CNMC_FRAME, 3)
    assert crop[0, 0].sum() == 0 and crop[CNMC_FRAME // 2, CNMC_FRAME // 2].sum() > 0
    # ruler mode keeps the cell-to-red-cell ratio: cell radius ~1.5 red-cell radii, red cell -> 157 px
    fg = (crop.max(axis=2) > 0).sum()
    d = 2 * np.sqrt(fg / np.pi)
    assert d == pytest.approx(1.5 * 157, rel=0.25)
