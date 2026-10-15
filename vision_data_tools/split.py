"""Split an image + YOLO-label folder into train / val / test and write ``data.yaml``."""
from __future__ import annotations

import random
import shutil
from pathlib import Path

from .io import list_images


def split_dataset(image_dir: str | Path, label_dir: str | Path, out_dir: str | Path, names: list[str],
                  ratios: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 0) -> dict[str, int]:
    """Copy images and labels into ``out/images/<split>`` and ``out/labels/<split>``.

    Images without a label file are kept as background images (empty label).
    """
    if abs(sum(ratios) - 1) > 1e-6:
        raise ValueError("ratios must sum to 1")
    images = list_images(image_dir)
    random.Random(seed).shuffle(images)
    n = len(images)
    n_train = round(n * ratios[0])
    n_val = round(n * ratios[1])
    parts = {"train": images[:n_train], "val": images[n_train:n_train + n_val], "test": images[n_train + n_val:]}

    out = Path(out_dir)
    for split, files in parts.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in files:
            shutil.copy2(img, out / "images" / split / img.name)
            lbl = Path(label_dir) / f"{img.stem}.txt"
            dst = out / "labels" / split / f"{img.stem}.txt"
            if lbl.exists():
                shutil.copy2(lbl, dst)
            else:
                dst.write_text("")

    yaml = [f"path: {out.resolve().as_posix()}", "train: images/train", "val: images/val", "test: images/test",
            f"nc: {len(names)}", "names:"] + [f"  {i}: {n}" for i, n in enumerate(names)]
    (out / "data.yaml").write_text("\n".join(yaml) + "\n")
    return {k: len(v) for k, v in parts.items()}
