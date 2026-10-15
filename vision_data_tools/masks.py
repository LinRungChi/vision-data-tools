"""Convert segmentation masks into YOLO segmentation labels (polygons).

Works with class-index masks (pixel value = class id, 0 = background) such as
those rendered from Blender, exported by labelling tools or produced by SAM.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .io import imread, list_images


def mask_to_polygons(mask: np.ndarray, class_ids: list[int] | None = None, min_area: float = 20,
                     simplify: float = 0.005) -> list[tuple[int, np.ndarray]]:
    """[(class_id, Nx2 polygon in pixels)] for every connected region of every class.

    ``simplify`` is the Douglas-Peucker tolerance as a fraction of the contour length.
    """
    ids = class_ids if class_ids is not None else [int(v) for v in np.unique(mask) if v != 0]
    out = []
    for c in ids:
        contours, _ = cv2.findContours((mask == c).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            if simplify > 0:
                cnt = cv2.approxPolyDP(cnt, simplify * cv2.arcLength(cnt, True), True)
            if len(cnt) >= 3:
                out.append((int(c), cnt.reshape(-1, 2)))
    return out


def polygons_to_yolo(polys: list[tuple[int, np.ndarray]], width: int, height: int,
                     remap: dict[int, int] | None = None) -> list[str]:
    """YOLO-seg lines: ``class x1 y1 x2 y2 ...`` normalised to [0, 1].

    ``remap`` converts mask values to training class ids (e.g. {1: 0, 2: 1}).
    """
    lines = []
    for c, poly in polys:
        c = remap.get(c, c) if remap else c
        xy = (poly / [width, height]).clip(0, 1).ravel()
        lines.append(f"{c} " + " ".join(f"{v:.6f}" for v in xy))
    return lines


def yolo_to_mask(lines: list[str], width: int, height: int) -> np.ndarray:
    """Inverse conversion, for visual checks."""
    mask = np.zeros((height, width), np.uint8)
    for line in lines:
        vals = line.split()
        pts = (np.array(vals[1:], float).reshape(-1, 2) * [width, height]).round().astype(np.int32)
        cv2.fillPoly(mask, [pts], int(vals[0]) or 255)
    return mask


def convert_folder(mask_dir: str | Path, label_dir: str | Path, remap: dict[int, int] | None = None,
                   **kwargs) -> int:
    """Convert every mask image in a folder; label files take the mask's file stem."""
    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in list_images(mask_dir):
        m = imread(p, cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        if m.ndim == 3:
            m = m[:, :, 0]
        lines = polygons_to_yolo(mask_to_polygons(m, **kwargs), m.shape[1], m.shape[0], remap)
        (label_dir / f"{p.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n += 1
    return n
