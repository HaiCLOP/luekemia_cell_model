"""Find white blood cells in a stained blood-smear image and cut each one out the way C-NMC crops look.

Smear (RGB) -> white balance -> background / cell mask -> stain-intensity map -> white-cell nuclei
-> split touching nuclei (watershed) -> add a thin cytoplasm rim -> one instance mask per white cell.

Then, for every cell: rescale the image so red blood cells have the diameter they would have at C-NMC
magnification (red cells are the ruler, about 7.5 um in any healthy smear), and paste the cell, centred,
onto a black 450 x 450 frame, which is what the classifier was trained on.

Everything is classical image processing (numpy, scipy, OpenCV, scikit-image): no training data, no GPU,
and it runs in the API. `method="cellpose"` swaps in Cellpose 3 for the cell outlines (research option,
needs torch + cellpose; not used by the API).

Assumptions, written down so they can be checked:
  * Romanowsky-type stain (Wright, Giemsa, May-Gruenwald-Giemsa): white-cell nuclei are the most
    saturated objects in the field after white balancing.
  * The field contains enough red cells to measure them (a normal peripheral smear does).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_multiotsu, threshold_otsu
from skimage.measure import regionprops
from skimage.segmentation import watershed

CNMC_FRAME = 450
# C-NMC normal lymphocytes have a median equivalent diameter of 180 px in their 450 px frames
# (reports/tables/val_cell_area.csv). A small lymphocyte is about 1.15 red-cell diameters across
# (~8.5 um vs ~7.5 um), so at C-NMC magnification a red cell would be about 180 / 1.15 = 157 px.
CNMC_RBC_PX = 157.0


@dataclass
class Cell:
    box: tuple[int, int, int, int]          # x0, y0, x1, y1 in the original image
    mask: np.ndarray                         # bool, original image size
    area_px: int
    nucleus_area_px: int
    stain: float                             # median stain intensity inside the nucleus (0-1)
    touches_border: bool
    lymphoid_like: bool = True               # one compact round nucleus filling most of the cell
    crop: np.ndarray | None = None           # 450 x 450 x 3 uint8, C-NMC style, filled in by to_cnmc
    extra: dict = field(default_factory=dict)


@dataclass
class SmearResult:
    cells: list[Cell]
    rbc_diameter_px: float | None
    scale: float | None                      # factor applied to reach C-NMC magnification
    notes: list[str]


# ----------------------------------------------------------------------------------- colour handling
def white_balance(rgb: np.ndarray) -> np.ndarray:
    """Divide by the background colour (brightest 10% of pixels), so a yellow or blue cast disappears."""
    f = rgb.astype(np.float32)
    lum = f.mean(axis=2)
    bg = f[lum >= np.percentile(lum, 90)].reshape(-1, 3).mean(axis=0)
    return np.clip(f / np.maximum(bg, 1.0) * 255.0, 0, 255).astype(np.uint8)


def maps(rgb_wb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stain intensity (0-1, white cells high) and optical density (any cell high)."""
    f = rgb_wb.astype(np.float32) / 255.0
    od = -np.log(np.clip(f, 1 / 255, 1)).sum(axis=2)
    hsv = cv2.cvtColor(rgb_wb, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[..., 1] / 255.0
    blue = np.clip(f[..., 2] - f[..., 0] * 0.5 - f[..., 1] * 0.5, 0, 1)   # purple/blue excess
    stain = np.clip(0.6 * sat + 0.8 * blue, 0, 1)
    return cv2.GaussianBlur(stain, (0, 0), 1.0), cv2.GaussianBlur(od, (0, 0), 1.0)


# ----------------------------------------------------------------------------------- red-cell ruler
def rbc_diameter(od: np.ndarray, exclude: np.ndarray | None = None, thr: float = 0.26, work: int = 256) -> float | None:
    """Typical red-cell diameter in pixels, by granulometry.

    Opening the cell mask with disks of growing radius removes objects narrower than the disk; the
    radius at which a lot of area disappears is a typical object radius. Touching red cells do not fool
    this the way connected components would. Clumps create a second, larger mode, so the first strong
    mode (at least 35% of the highest) is taken, not the global maximum.

    Checked against the BCCD red-cell boxes (src.smear_eval): see reports/smear/summary.json."""
    m = ndi.binary_fill_holes(od > thr)
    if exclude is not None:
        m &= ~exclude
    if m.mean() < 0.02:
        return None
    k = max(1.0, max(m.shape) / work)
    u8 = cv2.resize(m.astype(np.uint8), (max(1, int(m.shape[1] / k)), max(1, int(m.shape[0] / k))), interpolation=cv2.INTER_NEAREST)
    radii = np.arange(1, max(4, int(0.2 * min(u8.shape))))
    area = np.array([cv2.morphologyEx(u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))).sum()
                     for r in radii], dtype=np.float64)
    spec = -np.diff(area)
    if spec.sum() <= 0:
        return None
    spec = np.convolve(spec, [0.25, 0.5, 0.25], mode="same")
    rr = radii[1:]
    strong = [i for i in range(1, len(spec) - 1) if spec[i] >= spec[i - 1] and spec[i] >= spec[i + 1] and spec[i] >= 0.35 * spec.max()]
    i = strong[0] if strong else int(np.argmax(spec))
    lo, hi = max(0, i - 3), min(len(spec), i + 4)
    r = np.sum(rr[lo:hi] * spec[lo:hi]) / spec[lo:hi].sum()
    return float(2 * r * k)


# ----------------------------------------------------------------------------------- white cells
def find_white_cells(rgb: np.ndarray, method: str = "classic", min_nucleus_frac: float = 0.25,
                     rbc_override: float | None = None) -> SmearResult:
    """Instance masks for every white cell in the field.

    min_nucleus_frac: a nucleus must be at least this fraction of a red cell's area (drops platelets
    and stain debris).
    rbc_override: red-cell diameter measured elsewhere (e.g. pooled over all images from one microscope),
    used instead of this image's own estimate."""
    notes: list[str] = []
    wb = white_balance(rgb)
    stain, od = maps(wb)

    # cells vs background: Otsu on optical density, floored so an empty slide is not split
    # anything that is not background. White balance puts the background at OD ~0, so a fixed cut works
    # across microscopes; Otsu does not (on a pale field it splits white cells from everything else)
    fg = od > 0.18

    # nuclei: the top class of a 3-class Otsu on stain within the cells (background-ish / red cells /
    # white cells). No assumption about how many white cells there are: a leukemic field packed with
    # blasts must work as well as a normal one. A field with no white cells shows up as a top class that
    # is barely darker than the middle one, and is rejected by the gap test.
    vals = stain[fg]
    if vals.size < 200:
        return SmearResult([], None, None, notes + ["almost empty image"])
    t_mid, t_top = threshold_multiotsu(vals, classes=3)
    top, mid = vals[vals > t_top], vals[(vals > t_mid) & (vals <= t_top)]
    if top.size == 0 or mid.size == 0 or float(np.median(top) - np.median(mid)) < 0.12:
        d0 = rbc_override or rbc_diameter(od)
        return SmearResult([], d0, CNMC_RBC_PX / d0 if d0 else None, notes + ["no white cells found"])
    t_nuc = max(t_top, 0.25)
    nuclei = stain > t_nuc
    nuclei = ndi.binary_opening(nuclei, iterations=1)
    nuclei = ndi.binary_fill_holes(nuclei)

    d_rbc = rbc_override or rbc_diameter(od, exclude=ndi.binary_dilation(nuclei, iterations=3))
    if d_rbc is None:
        notes.append("too few red cells to measure scale")
    rbc_area = np.pi * (d_rbc / 2) ** 2 if d_rbc else None
    min_area = int(min_nucleus_frac * rbc_area) if rbc_area else 30

    lab, n = ndi.label(nuclei)
    if n == 0:
        return SmearResult([], d_rbc, CNMC_RBC_PX / d_rbc if d_rbc else None, notes + ["no white cells found"])
    sizes = ndi.sum(nuclei, lab, index=np.arange(1, n + 1))
    # lobes of one nucleus are each small; keep them for now and merge before the size filter
    keep = np.isin(lab, np.flatnonzero(sizes >= 0.08 * (rbc_area or 400)) + 1)
    keep = _merge_lobes(keep, d_rbc, rbc_area)
    lab2, n2 = ndi.label(keep, structure=np.ones((3, 3)))
    sizes2 = ndi.sum(keep, lab2, index=np.arange(1, n2 + 1))
    keep = np.isin(lab2, np.flatnonzero(sizes2 >= min_area) + 1)
    if not keep.any():
        return SmearResult([], d_rbc, CNMC_RBC_PX / d_rbc if d_rbc else None, notes + ["no white cells found"])

    if method == "cellpose":
        cells_lab = _cellpose_cells(rgb, d_rbc, keep)
    else:
        cells_lab = _split_and_grow(keep, fg, d_rbc)
        cells_lab = _group_fragments(cells_lab, keep, d_rbc)

    cells = []
    H, W = keep.shape
    for rp in regionprops(cells_lab):
        m = cells_lab == rp.label
        nuc = m & keep
        if nuc.sum() < min_area:
            continue
        y0, x0, y1, x1 = rp.bbox
        nlab, nparts = ndi.label(nuc)
        solidity = regionprops(nuc.astype(np.uint8))[0].solidity
        lymphoid = nparts == 1 and solidity >= 0.88
        cells.append(Cell(box=(int(x0), int(y0), int(x1), int(y1)), mask=m, area_px=int(m.sum()),
                          nucleus_area_px=int(nuc.sum()), stain=float(np.median(stain[nuc])),
                          touches_border=bool(x0 == 0 or y0 == 0 or x1 >= W or y1 >= H),
                          lymphoid_like=bool(lymphoid), extra={"nucleus_parts": int(nparts), "solidity": float(solidity)}))
    return SmearResult(cells, d_rbc, CNMC_RBC_PX / d_rbc if d_rbc else None, notes)


def _merge_lobes(nuclei: np.ndarray, d_rbc: float | None, rbc_area: float | None) -> np.ndarray:
    """Bridge small nucleus fragments that sit close together (lobes of one neutrophil, a nucleus cut by
    a pale band). Two large nuclei next to each other are left apart: those are two cells."""
    if not d_rbc:
        return nuclei
    lab, n = ndi.label(nuclei)
    if n < 2:
        return nuclei
    sizes = ndi.sum(nuclei, lab, index=np.arange(1, n + 1))
    small = np.isin(lab, np.flatnonzero(sizes < 0.7 * rbc_area) + 1)
    gap = max(1, int(0.20 * d_rbc))
    bridged = ndi.binary_closing(small, structure=np.ones((3, 3)), iterations=gap) & ndi.binary_dilation(small, iterations=gap)
    return nuclei | bridged


def _group_fragments(cells_lab: np.ndarray, nuclei: np.ndarray, d_rbc: float | None) -> np.ndarray:
    """Join cells whose nuclei sit within a quarter red-cell diameter of each other, unless both are
    large round nuclei (two blasts side by side). A joined cell has a multi-part nucleus, so it is
    marked non-lymphoid and not scored: the lobes of a neutrophil must not pass as small lymphocytes."""
    if not d_rbc or cells_lab.max() < 2:
        return cells_lab
    rbc_area = np.pi * (d_rbc / 2) ** 2
    gap = max(1, int(0.25 * d_rbc))
    ids = [i for i in np.unique(cells_lab) if i]
    blastlike, near = {}, {}
    for i in ids:
        nuc = (cells_lab == i) & nuclei
        if not nuc.any():
            blastlike[i] = False
            continue
        rp = regionprops(nuc.astype(np.uint8))[0]
        blastlike[i] = bool(rp.area >= 0.5 * rbc_area and rp.solidity >= 0.9)
        y0, x0, y1, x1 = rp.bbox
        ys, xs = slice(max(0, y0 - gap), y1 + gap), slice(max(0, x0 - gap), x1 + gap)
        grown = ndi.binary_dilation(nuc[ys, xs], iterations=gap)
        near[i] = set(np.unique(cells_lab[ys, xs][grown & nuclei[ys, xs]])) - {0, i}
    parent = {i: i for i in ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in ids:
        for j in near.get(i, ()):
            if not (blastlike[i] and blastlike.get(j, False)):
                parent[find(i)] = find(j)
    out = np.zeros_like(cells_lab)
    for i in ids:
        out[cells_lab == i] = find(i)
    return out


def _split_and_grow(nuclei: np.ndarray, fg: np.ndarray, d_rbc: float | None) -> np.ndarray:
    """Watershed on the nucleus distance map splits touching nuclei; each cell then gets a thin
    cytoplasm rim (white-cell pixels around the nucleus, limited to ~12% of the nucleus diameter)."""
    comp, n = ndi.label(nuclei, structure=np.ones((3, 3)))
    nuc_lab = np.zeros_like(comp)
    nxt = 1
    rbc_area = np.pi * (d_rbc / 2) ** 2 if d_rbc else None
    for i in range(1, n + 1):
        m = comp == i
        area = m.sum()
        if rbc_area and area > 2.2 * rbc_area:      # big enough to be two touching cells: try to split
            dist = ndi.distance_transform_edt(m)
            pk = peak_local_max(dist, min_distance=int(max(4, 0.45 * d_rbc)), labels=m.astype(int), exclude_border=False)
            if len(pk) > 1:
                mk = np.zeros_like(comp)
                for j, (y, x) in enumerate(pk, 1):
                    mk[y, x] = j
                ws = watershed(-dist, mk, mask=m)
                parts = [ws == j for j in range(1, ws.max() + 1)]
                # accept the split only if every piece is a round nucleus of a plausible size: touching
                # blasts split, while the lobes of a neutrophil or eosinophil stay together as one cell
                ok = all(p.sum() >= 0.5 * rbc_area and regionprops(p.astype(np.uint8))[0].solidity >= 0.9 for p in parts)
                if ok:
                    for p in parts:
                        nuc_lab[p] = nxt
                        nxt += 1
                    continue
        nuc_lab[m] = nxt
        nxt += 1
    out = np.zeros_like(nuc_lab)
    for rp in regionprops(nuc_lab):
        rim = max(1, int(0.12 * rp.equivalent_diameter_area))
        y0, x0, y1, x1 = rp.bbox
        pad = rim + 2
        ys, xs = slice(max(0, y0 - pad), y1 + pad), slice(max(0, x0 - pad), x1 + pad)
        local = nuc_lab[ys, xs] == rp.label
        grown = ndi.binary_dilation(local, iterations=rim) & (fg[ys, xs] | local)
        region = out[ys, xs]
        region[grown & (region == 0)] = rp.label
    return out


def _cellpose_cells(rgb: np.ndarray, d_rbc: float | None, nuclei: np.ndarray) -> np.ndarray:
    """Cellpose 3 "cyto3" outlines for every cell; keep those that contain a white-cell nucleus.
    Research comparison only (recall 82.5% vs 98.4% for the classical finder on BCCD, 12x slower). cellpose<4
    needs numpy<2.1, so run it in a separate environment from the pinned one."""
    from cellpose import models  # optional dependency

    global _CP
    try:
        _CP
    except NameError:
        _CP = models.Cellpose(gpu=True, model_type="cyto3")
    masks, *_ = _CP.eval(rgb, diameter=(1.3 * d_rbc) if d_rbc else None, channels=[0, 0])
    out = np.zeros_like(masks)
    k = 1
    for lab_id in np.unique(masks[nuclei]):
        if lab_id == 0:
            continue
        m = masks == lab_id
        if (m & nuclei).sum() >= 0.2 * m.sum():
            out[m] = k
            k += 1
    return out


# ----------------------------------------------------------------------------------- C-NMC style crops
def to_cnmc(rgb: np.ndarray, cell: Cell, scale: float | None, mode: str = "ruler", stain: dict | None = None) -> np.ndarray:
    """Cut the cell out onto black and place it in a 450 x 450 frame.

    mode="ruler":  rescale by `scale` (red cells -> C-NMC size). Keeps true relative cell size.
    mode="fit":    rescale so the cell is as large as a median C-NMC normal cell. Removes absolute size.
    mode="native": no rescaling (the naive approach; included only as a comparison).
    stain: C-NMC colour statistics; when given, the cell's colours are normalised to them."""
    x0, y0, x1, y1 = cell.box
    pad = 4
    xa, ya, xb, yb = max(0, x0 - pad), max(0, y0 - pad), min(rgb.shape[1], x1 + pad), min(rgb.shape[0], y1 + pad)
    patch = rgb[ya:yb, xa:xb].copy()
    m = cell.mask[ya:yb, xa:xb]
    patch[~m] = 0
    if mode == "ruler" and scale:
        f = scale
    elif mode == "fit" or (mode == "ruler" and not scale):
        f = 180.4 / max(1.0, 2 * np.sqrt(cell.area_px / np.pi))
    else:
        f = 1.0
    h, w = patch.shape[:2]
    nh, nw = max(1, int(round(h * f))), max(1, int(round(w * f)))
    interp = cv2.INTER_CUBIC if f > 1 else cv2.INTER_AREA
    patch = cv2.resize(patch, (nw, nh), interpolation=interp)
    mk = cv2.resize(m.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST).astype(bool)
    patch[~mk] = 0
    frame = np.zeros((CNMC_FRAME, CNMC_FRAME, 3), np.uint8)
    # centre on the mask centroid; crop anything that would fall outside the frame
    cy, cx = [int(round(v)) for v in ndi.center_of_mass(mk)] if mk.any() else (nh // 2, nw // 2)
    oy, ox = CNMC_FRAME // 2 - cy, CNMC_FRAME // 2 - cx
    sy0, sx0 = max(0, -oy), max(0, -ox)
    dy0, dx0 = max(0, oy), max(0, ox)
    hh, ww = min(nh - sy0, CNMC_FRAME - dy0), min(nw - sx0, CNMC_FRAME - dx0)
    if hh > 0 and ww > 0:
        frame[dy0:dy0 + hh, dx0:dx0 + ww] = patch[sy0:sy0 + hh, sx0:sx0 + ww]
    return stain_normalise(frame, stain) if stain else frame


def stain_normalise(crop: np.ndarray, target: dict) -> np.ndarray:
    """Reinhard colour transfer on the cell pixels only: match the mean and spread of each LAB channel to
    the C-NMC training cells (models/cnmc_stain.json). The black background is left black.

    C-NMC cells are dark maroon-purple; other labs' smears are often bright blue-violet or pink. Without
    this step the classifier sees colours it never trained on (on healthy BCCD lymphocytes it called 92%
    of them leukemic)."""
    m = crop.max(axis=2) > 15
    if m.sum() < 50:
        return crop
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    px = lab[m]
    mu, sd = px.mean(0), px.std(0) + 1e-3
    tm, ts = np.array(target["mean"], np.float32), np.array(target["std"], np.float32)
    lab[m] = (px - mu) / sd * ts + tm
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    out[~m] = 0
    return out


def analyse(rgb: np.ndarray, method: str = "classic", mode: str = "ruler", rbc_override: float | None = None,
            stain: dict | None = None) -> SmearResult:
    res = find_white_cells(rgb, method=method, rbc_override=rbc_override)
    for c in res.cells:
        c.crop = to_cnmc(rgb, c, res.scale, mode=mode, stain=stain)
    return res
