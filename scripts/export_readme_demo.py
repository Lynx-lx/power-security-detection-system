"""导出 README 用的检测效果图（白框引线 + 风险条）。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from infer_expected_style import draw_expected_style, imread_bgr, imwrite_bgr
from viscale.detection.yolov5s_lite import POWER_SECURITY_CLASSES
from viscale.risk import DetectionRecord, RiskAssessor

WEIGHTS = ROOT / "models" / "checkpoints" / "yolov8n_power.pt"
OUT = ROOT / "assets" / "demo" / "infer_result.png"
MAX_W = 1280


def resize_max_w(img: np.ndarray, max_w: int = MAX_W) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / float(w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def add_risk_banner(bgr: np.ndarray, label: str, score: float, n_det: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    bar_h = 56
    canvas = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    canvas[bar_h:] = bgr
    canvas[:bar_h] = (36, 36, 36)
    text = f"Risk: {label}  |  score {score:.2f}  |  boxes {n_det}"
    cv2.putText(canvas, text, (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def main() -> None:
    from ultralytics import YOLO

    src = ROOT / "data" / "samples" / "scene2_orig.png"
    if not src.is_file():
        raise FileNotFoundError(src)
    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)

    image = imread_bgr(src)
    model = YOLO(str(WEIGHTS))
    result = model.predict(image, conf=0.25, iou=0.45, imgsz=640, device="cpu", verbose=False)[0]
    names = list(POWER_SECURITY_CLASSES)
    rows: list[dict] = []
    records: list[DetectionRecord] = []
    if result.boxes is not None:
        for box, score, cid in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy().astype(int),
        ):
            cid_i = int(cid)
            name = names[cid_i] if 0 <= cid_i < len(names) else str(cid_i)
            xyxy = [int(round(v)) for v in box.tolist()]
            rows.append({"cls_name": name, "conf": float(score), "xyxy": xyxy})
            records.append(DetectionRecord(cls_name=name, conf=float(score), xyxy=tuple(float(v) for v in xyxy)))

    vis = draw_expected_style(image, rows)
    h, w = vis.shape[:2]
    report = RiskAssessor().assess(records, meters_per_pixel=None, image_wh=(w, h), update_elc=False)
    vis = add_risk_banner(vis, report.label_zh, report.score, len(rows))
    vis = resize_max_w(vis)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    imwrite_bgr(OUT, vis)
    print(f"detections={len(rows)} grade={report.label_zh} saved={OUT}")


if __name__ == "__main__":
    main()
