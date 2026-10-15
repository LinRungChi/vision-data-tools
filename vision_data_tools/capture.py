"""Record image sequences from an industrial (Basler / pylon) camera or any OpenCV camera.

    vdt capture --source pylon --out recordings --interval 0.5
    vdt capture --source 0     --out recordings            # webcam

Keys in the preview window: ``r`` start/stop recording, ``s`` save one frame, ``q`` quit.
Every run writes to a new ``<out>/<date>/session_<n>`` folder.
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class OpenCVSource:
    def __init__(self, index: int = 0):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {index}")

    def read(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self) -> None:
        self.cap.release()


class PylonSource:
    """First Basler camera found by pylon; frames are returned as BGR."""

    def __init__(self):
        from pypylon import pylon  # optional dependency: pip install pypylon
        self.pylon = pylon
        self.cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.cam.Open()
        self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.conv = pylon.ImageFormatConverter()
        self.conv.OutputPixelFormat = pylon.PixelType_BGR8packed

    def read(self) -> np.ndarray | None:
        res = self.cam.RetrieveResult(5000, self.pylon.TimeoutHandling_Return)
        try:
            return self.conv.Convert(res).GetArray() if res and res.GrabSucceeded() else None
        finally:
            if res:
                res.Release()

    def close(self) -> None:
        if self.cam.IsGrabbing():
            self.cam.StopGrabbing()
        self.cam.Close()


def pylon_camera_info() -> list[dict]:
    """Model, resolution, pixel format and frame-rate limits of every connected Basler camera."""
    from pypylon import pylon
    tl = pylon.TlFactory.GetInstance()
    info = []
    for dev in tl.EnumerateDevices():
        cam = pylon.InstantCamera(tl.CreateDevice(dev))
        cam.Open()
        d = {"model": dev.GetModelName(), "serial": dev.GetSerialNumber()}
        for key in ("Width", "Height", "PixelFormat", "ResultingFrameRate", "ResultingFrameRateAbs",
                    "AcquisitionFrameRate", "AcquisitionFrameRateAbs"):
            try:
                d[key] = getattr(cam, key).GetValue()
            except Exception:  # node not available on this model
                pass
        cam.Close()
        info.append(d)
    return info


def new_session_dir(root: str | Path) -> Path:
    day = Path(root) / datetime.now().strftime("%Y-%m-%d")
    n = 1
    while (day / f"session_{n}").exists():
        n += 1
    path = day / f"session_{n}"
    path.mkdir(parents=True)
    return path


def record(source, out_root: str | Path, interval: float = 0.0, preview_width: int = 960,
           start_recording: bool = False, max_frames: int | None = None, show: bool = True) -> Path:
    """Preview and record frames. ``interval`` = minimum seconds between saved frames."""
    out = new_session_dir(out_root)
    recording, last, saved = start_recording, 0.0, 0
    try:
        while max_frames is None or saved < max_frames:
            frame = source.read()
            if frame is None:
                if not show:
                    break
                continue
            now = time.time()
            if recording and now - last >= interval:
                cv2.imwrite(str(out / f"img_{saved:05d}.jpg"), frame)
                saved, last = saved + 1, now
            if show:
                scale = preview_width / frame.shape[1]
                vis = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame.copy()
                cv2.putText(vis, f"{'REC' if recording else 'preview'}  saved {saved}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if recording else (0, 255, 0), 2)
                cv2.imshow("capture", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    recording = not recording
                if key == ord("s"):
                    cv2.imwrite(str(out / f"img_{saved:05d}.jpg"), frame)
                    saved += 1
    finally:
        source.close()
        if show:
            cv2.destroyAllWindows()
    print(f"{saved} frames saved to {out}")
    return out
