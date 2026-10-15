"""Build docs/overview.jpg from synthetic images (no real data needed)."""
from pathlib import Path

import cv2
import numpy as np

from vision_data_tools.autolabel import ThresholdLabeler
from vision_data_tools.count import ParticleCounter
from vision_data_tools.masks import mask_to_polygons

rng = np.random.default_rng(3)
H, W = 300, 400


def title(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (W, 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# 1. back-lit tank with small dark objects -> auto boxes
tank = np.full((H, W), 210, np.float32) - 30 * np.linspace(0, 1, W)[None, :]
for _ in range(14):
    c = (int(rng.integers(30, W - 30)), int(rng.integers(40, H - 20)))
    cv2.ellipse(tank, c, (int(rng.integers(8, 18)), int(rng.integers(3, 6))), float(rng.uniform(0, 180)), 0, 360, 40, -1)
tank = cv2.GaussianBlur(tank + rng.normal(0, 4, tank.shape), (3, 3), 0).clip(0, 255).astype(np.uint8)
tank = cv2.cvtColor(tank, cv2.COLOR_GRAY2BGR)
lab = ThresholdLabeler(threshold=110, min_area=30)
a = tank.copy()
for x, y, w, h in lab.boxes(tank):
    cv2.rectangle(a, (x, y), (x + w, y + h), (0, 200, 0), 2)
a = title(a, f"autolabel: {len(lab.boxes(tank))} YOLO boxes")

# 2. class-index mask -> polygons
mask = np.zeros((H, W), np.uint8)
cv2.ellipse(mask, (140, 170), (90, 60), 20, 0, 360, 1, -1)
cv2.ellipse(mask, (290, 140), (50, 80), -30, 0, 360, 2, -1)
colours = {1: (180, 120, 60), 2: (60, 140, 220)}
b = np.full((H, W, 3), 245, np.uint8)
for c, col in colours.items():
    b[mask == c] = col
for c, poly in mask_to_polygons(mask, simplify=0.01):
    cv2.polylines(b, [poly.astype(np.int32)], True, (0, 0, 0), 2)
    for x, y in poly:
        cv2.circle(b, (int(x), int(y)), 3, (0, 0, 255), -1)
b = title(b, "masks2yolo: mask -> polygon labels")

# 3. microscopy particles on uneven background
micro = (70 + 60 * np.linspace(0, 1, W)[None, :] + 25 * np.linspace(1, 0, H)[:, None]
         + rng.normal(0, 3, (H, W))).clip(0, 255).astype(np.uint8)
for _ in range(60):
    cv2.circle(micro, (int(rng.integers(10, W - 10)), int(rng.integers(35, H - 10))), int(rng.integers(2, 4)), 235, -1)
cv2.ellipse(micro, (300, 220), (35, 20), 10, 0, 360, 230, -1)  # debris
n, c3 = ParticleCounter(tophat_kernel=15, min_area=5, max_area=100).annotate(cv2.cvtColor(micro, cv2.COLOR_GRAY2BGR))
c3 = title(c3, f"count: top-hat + Otsu + area filter ({n})")

Path("docs").mkdir(exist_ok=True)
cv2.imwrite("docs/overview.jpg", np.hstack([a, np.full((H, 6, 3), 255, np.uint8), b, np.full((H, 6, 3), 255, np.uint8), c3]),
            [cv2.IMWRITE_JPEG_QUALITY, 90])
print("docs/overview.jpg")
