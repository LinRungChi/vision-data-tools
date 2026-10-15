"""Bootstrap detection labels for objects that are darker (or brighter) than their background.

Typical for back-lit setups: fish fry or particles in a tank, parts on a light
table, cells in bright-field microscopy. Threshold -> morphological clean-up ->
connected regions -> YOLO boxes. The labels are a starting point to be
reviewed, not ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .io import imread, imwrite, list_images


@dataclass
class ThresholdLabeler:
    threshold: int = 50          # grey level separating object from background
    dark_objects: bool = True    # False: objects brighter than background
    min_area: int = 100          # px, smaller regions are noise
    max_area: int | None = None
    kernel: int = 5              # morphology kernel size
    class_id: int = 0

    def mask(self, img: np.ndarray) -> np.ndarray:
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        m = (gray < self.threshold) if self.dark_objects else (gray > self.threshold)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel, self.kernel))
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, k)
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, k)

    def boxes(self, img: np.ndarray) -> list[tuple[int, int, int, int]]:
        """(x, y, w, h) of every object region."""
        n, _, stats, _ = cv2.connectedComponentsWithStats(self.mask(img), connectivity=8)
        out = []
        for x, y, w, h, area in stats[1:]:
            if area >= self.min_area and (self.max_area is None or area <= self.max_area):
                out.append((int(x), int(y), int(w), int(h)))
        return out

    def yolo_lines(self, img: np.ndarray) -> list[str]:
        H, W = img.shape[:2]
        return [f"{self.class_id} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}"
                for x, y, w, h in self.boxes(img)]

    def label_folder(self, image_dir: str | Path, label_dir: str | Path, preview_dir: str | Path | None = None,
                     skip_empty: bool = False) -> dict[str, int]:
        """Write one YOLO ``.txt`` per image. Returns {image name: number of boxes}."""
        label_dir = Path(label_dir)
        label_dir.mkdir(parents=True, exist_ok=True)
        if preview_dir:
            Path(preview_dir).mkdir(parents=True, exist_ok=True)
        counts = {}
        for p in list_images(image_dir):
            img = imread(p)
            if img is None:
                continue
            lines = self.yolo_lines(img)
            if skip_empty and not lines:
                continue
            (label_dir / f"{p.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            counts[p.name] = len(lines)
            if preview_dir:
                vis = img.copy()
                for x, y, w, h in self.boxes(img):
                    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                imwrite(Path(preview_dir) / p.name, vis)
        return counts
