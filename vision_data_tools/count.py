"""Count small bright particles (cells, algae, beads) in microscopy frames.

White top-hat removes uneven illumination, Otsu picks the threshold, a small
opening removes speckle, and an area window rejects debris.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class ParticleCounter:
    tophat_kernel: int = 15   # a bit larger than the biggest particle, in px
    open_kernel: int = 3
    min_area: int = 5
    max_area: int = 100
    dark_particles: bool = False  # True: particles darker than background (black top-hat)

    def mask(self, img: np.ndarray) -> np.ndarray:
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.tophat_kernel, self.tophat_kernel))
        th = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT if self.dark_particles else cv2.MORPH_TOPHAT, k)
        _, binary = cv2.threshold(th, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if self.open_kernel > 1:
            ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.open_kernel, self.open_kernel))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ko)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        keep = np.zeros(n, bool)
        keep[1:] = (stats[1:, cv2.CC_STAT_AREA] >= self.min_area) & (stats[1:, cv2.CC_STAT_AREA] <= self.max_area)
        return (keep[labels] * 255).astype(np.uint8)

    def count(self, img: np.ndarray) -> tuple[int, np.ndarray]:
        """(number of particles, filtered binary mask)."""
        m = self.mask(img)
        n, _ = cv2.connectedComponents(m, connectivity=8)
        return n - 1, m

    def annotate(self, img: np.ndarray) -> tuple[int, np.ndarray]:
        n, m = self.count(img)
        vis = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 1)
        cv2.putText(vis, f"count: {n}", (12, vis.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        return n, vis

    def count_video(self, video: str | Path, csv_path: str | Path, out_video: str | Path | None = None) -> list[int]:
        """Per-frame counts to CSV (frame, time_s, count); optional annotated side-by-side video."""
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise FileNotFoundError(video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        writer, counts = None, []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n, vis = self.annotate(frame)
            counts.append(n)
            if out_video:
                side = np.hstack([vis, cv2.cvtColor(self.mask(frame), cv2.COLOR_GRAY2BGR)])
                if writer is None:
                    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                             (side.shape[1], side.shape[0]))
                writer.write(side)
        cap.release()
        if writer:
            writer.release()
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "time_s", "count"])
            w.writerows((i, round(i / fps, 3), c) for i, c in enumerate(counts))
        return counts
