import cv2
import numpy as np

from vision_data_tools.autolabel import ThresholdLabeler
from vision_data_tools.capture import record
from vision_data_tools.count import ParticleCounter
from vision_data_tools.io import imread, imwrite
from vision_data_tools.masks import convert_folder, mask_to_polygons, polygons_to_yolo, yolo_to_mask
from vision_data_tools.split import split_dataset


def _scene():
    img = np.full((200, 300, 3), 220, np.uint8)
    cv2.circle(img, (60, 60), 20, (20, 20, 20), -1)
    cv2.rectangle(img, (150, 100), (200, 160), (30, 30, 30), -1)
    cv2.circle(img, (260, 30), 2, (10, 10, 10), -1)  # speck, below min_area
    return img


def test_autolabel_finds_dark_objects_and_normalises():
    lab = ThresholdLabeler(threshold=100, min_area=50)
    boxes = sorted(lab.boxes(_scene()))
    assert len(boxes) == 2
    x, y, w, h = boxes[0]
    assert abs(x - 40) <= 2 and abs(w - 41) <= 3
    cls, cx, cy, bw, bh = map(float, lab.yolo_lines(_scene())[0].split())
    assert cls == 0 and 0 < cx < 1 and 0 < bh < 1


def test_autolabel_folder_with_unicode_path(tmp_path):
    d = tmp_path / "影像"
    d.mkdir()
    imwrite(d / "a.png", _scene())
    imwrite(d / "b.png", np.full((50, 50, 3), 220, np.uint8))
    counts = ThresholdLabeler(threshold=100, min_area=50).label_folder(d, tmp_path / "labels", tmp_path / "prev")
    assert counts == {"a.png": 2, "b.png": 0}
    assert (tmp_path / "labels" / "b.txt").read_text() == ""
    assert imread(tmp_path / "prev" / "a.png") is not None


def test_mask_polygons_roundtrip(tmp_path):
    mask = np.zeros((100, 120), np.uint8)
    cv2.rectangle(mask, (10, 10), (50, 60), 1, -1)
    cv2.circle(mask, (90, 50), 15, 2, -1)
    polys = mask_to_polygons(mask)
    assert sorted(c for c, _ in polys) == [1, 2]
    lines = polygons_to_yolo(polys, 120, 100, remap={1: 0, 2: 1})
    assert sorted(l.split()[0] for l in lines) == ["0", "1"]
    back = yolo_to_mask([l for l in lines if l.startswith("1 ")], 120, 100)
    inter = ((back > 0) & (mask == 2)).sum() / (mask == 2).sum()
    assert inter > 0.9
    imwrite(tmp_path / "m.png", mask)
    assert convert_folder(tmp_path, tmp_path / "lbl") == 1


def test_split_writes_yaml_and_background_labels(tmp_path):
    imgs, lbls = tmp_path / "i", tmp_path / "l"
    imgs.mkdir()
    lbls.mkdir()
    for i in range(10):
        imwrite(imgs / f"{i}.png", np.zeros((8, 8, 3), np.uint8))
        if i % 2:
            (lbls / f"{i}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    n = split_dataset(imgs, lbls, tmp_path / "ds", ["fry"], (0.6, 0.2, 0.2))
    assert n == {"train": 6, "val": 2, "test": 2}
    assert "0: fry" in (tmp_path / "ds" / "data.yaml").read_text()
    assert len(list((tmp_path / "ds" / "labels").rglob("*.txt"))) == 10


def test_particle_counter_counts_and_rejects_debris():
    rng = np.random.default_rng(0)
    img = (60 + 40 * np.linspace(0, 1, 400)[None, :] + rng.normal(0, 2, (300, 400))).clip(0, 255).astype(np.uint8)
    pts = [(int(x), int(y)) for x, y in zip(rng.integers(20, 380, 25), rng.integers(20, 280, 25))]
    for x, y in pts:
        cv2.circle(img, (x, y), 3, 255, -1)
    cv2.circle(img, (200, 150), 25, 255, -1)  # debris, too large
    n, _ = ParticleCounter(tophat_kernel=15, min_area=5, max_area=100).count(img)
    assert 20 <= n <= 25  # overlapping dots may merge


class _FakeSource:
    def __init__(self, n):
        self.n = n

    def read(self):
        if self.n == 0:
            return None
        self.n -= 1
        return np.zeros((20, 20, 3), np.uint8)

    def close(self):
        pass


def test_record_headless(tmp_path):
    out = record(_FakeSource(5), tmp_path, start_recording=True, show=False)
    assert len(list(out.glob("*.jpg"))) == 5
    assert out.name == "session_1"
