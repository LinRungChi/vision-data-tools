"""``vdt`` command-line interface."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _remap(s: str | None) -> dict[int, int] | None:
    """'1:0,2:1' -> {1: 0, 2: 1}"""
    if not s:
        return None
    return {int(a): int(b) for a, b in (kv.split(":") for kv in s.split(","))}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="vdt", description="vision dataset tools")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("autolabel", help="threshold-based YOLO box labels for dark/bright objects")
    a.add_argument("images", type=Path)
    a.add_argument("labels", type=Path)
    a.add_argument("--threshold", type=int, default=50)
    a.add_argument("--bright", action="store_true", help="objects are brighter than the background")
    a.add_argument("--min-area", type=int, default=100)
    a.add_argument("--max-area", type=int)
    a.add_argument("--class-id", type=int, default=0)
    a.add_argument("--preview", type=Path, help="folder for images with boxes drawn")

    m = sub.add_parser("masks2yolo", help="class-index masks -> YOLO segmentation polygons")
    m.add_argument("masks", type=Path)
    m.add_argument("labels", type=Path)
    m.add_argument("--remap", help="mask value to class id, e.g. '1:0,2:1'")
    m.add_argument("--min-area", type=float, default=20)
    m.add_argument("--simplify", type=float, default=0.005)

    s = sub.add_parser("split", help="train/val/test split + data.yaml")
    s.add_argument("images", type=Path)
    s.add_argument("labels", type=Path)
    s.add_argument("out", type=Path)
    s.add_argument("--names", required=True, help="comma-separated class names")
    s.add_argument("--ratios", default="0.8,0.1,0.1")
    s.add_argument("--seed", type=int, default=0)

    c = sub.add_parser("count", help="count small particles per frame of a microscopy video")
    c.add_argument("video", type=Path)
    c.add_argument("--csv", type=Path, default=Path("counts.csv"))
    c.add_argument("--out-video", type=Path)
    c.add_argument("--kernel", type=int, default=15)
    c.add_argument("--min-area", type=int, default=5)
    c.add_argument("--max-area", type=int, default=100)
    c.add_argument("--dark", action="store_true")

    r = sub.add_parser("capture", help="record frames from a Basler (pylon) or OpenCV camera")
    r.add_argument("--source", default="pylon", help="'pylon' or an OpenCV camera index")
    r.add_argument("--out", type=Path, default=Path("recordings"))
    r.add_argument("--interval", type=float, default=0.0)

    sub.add_parser("caminfo", help="list connected Basler cameras and their limits")

    args = p.parse_args(argv)
    if args.cmd == "autolabel":
        from .autolabel import ThresholdLabeler
        lab = ThresholdLabeler(threshold=args.threshold, dark_objects=not args.bright, min_area=args.min_area,
                               max_area=args.max_area, class_id=args.class_id)
        counts = lab.label_folder(args.images, args.labels, args.preview)
        print(f"{len(counts)} images, {sum(counts.values())} boxes, {sum(v == 0 for v in counts.values())} without objects")
    elif args.cmd == "masks2yolo":
        from .masks import convert_folder
        print(convert_folder(args.masks, args.labels, _remap(args.remap), min_area=args.min_area,
                             simplify=args.simplify), "masks converted")
    elif args.cmd == "split":
        from .split import split_dataset
        ratios = tuple(float(x) for x in args.ratios.split(","))
        print(split_dataset(args.images, args.labels, args.out, args.names.split(","), ratios, args.seed))
    elif args.cmd == "count":
        from .count import ParticleCounter
        pc = ParticleCounter(tophat_kernel=args.kernel, min_area=args.min_area, max_area=args.max_area,
                             dark_particles=args.dark)
        counts = pc.count_video(args.video, args.csv, args.out_video)
        print(f"{len(counts)} frames, mean count {sum(counts) / max(len(counts), 1):.1f} -> {args.csv}")
    elif args.cmd == "capture":
        from .capture import OpenCVSource, PylonSource, record
        src = PylonSource() if args.source == "pylon" else OpenCVSource(int(args.source))
        record(src, args.out, args.interval)
    elif args.cmd == "caminfo":
        from .capture import pylon_camera_info
        print(json.dumps(pylon_camera_info(), indent=1, default=str))


if __name__ == "__main__":
    main()
