"""按预期效果图风格画白框 + 引出线类别名，并用公开数据微调后的 Ultralytics 权重推理。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from viscale.detection.yolov5s_lite import POWER_SECURITY_CLASSES

DISPLAY_NAME = {
    "insulator": "insulator",
    "bird_nest": "nest",
    "foreign_object": "foreign_object",
    "damaged_insulator": "damaged_insulator",
}


def imread_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def imwrite_bgr(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise OSError(path)
    buf.tofile(str(path))


def draw_expected_style(image: np.ndarray, rows: list[dict]) -> np.ndarray:
    vis = image.copy()
    h, w = vis.shape[:2]
    used_slots: list[tuple[int, int, int, int]] = []

    def place_label(px: int, py: int, tw: int, th: int) -> tuple[int, int]:
        candidates = [
            (px + 24, py - th - 8),
            (px - tw - 24, py - th - 8),
            (px + 24, py + 12),
            (px - tw - 24, py + 12),
            (min(w - tw - 6, px + 40), max(6, py - 80)),
        ]
        for x, y in candidates:
            x = int(np.clip(x, 4, max(4, w - tw - 4)))
            y = int(np.clip(y, 4, max(4, h - th - 4)))
            box = (x, y, x + tw, y + th)
            overlap = False
            for u in used_slots:
                if not (box[2] < u[0] or box[0] > u[2] or box[3] < u[1] or box[1] > u[3]):
                    overlap = True
                    break
            if not overlap:
                used_slots.append(box)
                return x, y
        used_slots.append((px, py, px + tw, py + th))
        return px, py

    for row in rows:
        x1, y1, x2, y2 = (int(v) for v in row["xyxy"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 2)
        name = DISPLAY_NAME.get(row["cls_name"], row["cls_name"])
        text = f"{name}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cx = (x1 + x2) // 2
        cy = y1
        lx, ly = place_label(cx, cy, tw + 8, th + 10)
        ax, ay = cx, y1
        cv2.line(vis, (ax, ay), (lx, ly + th // 2), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, text, (lx, ly + th), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--weights", type=str, default="models/checkpoints/yolov8n_power.pt")
    p.add_argument("--out", type=str, default="outputs/detections/scene2_expected_style.jpg")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--imgsz", type=int, default=640)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = ROOT / weights
    image = imread_bgr(Path(args.image) if Path(args.image).is_absolute() else ROOT / args.image)
    model = YOLO(str(weights))
    result = model.predict(image, conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=args.device, verbose=False)[0]
    rows: list[dict] = []
    names = list(POWER_SECURITY_CLASSES)
    if result.boxes is not None:
        for box, score, cid in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy().astype(int),
        ):
            cid = int(cid)
            name = names[cid] if 0 <= cid < len(names) else str(cid)
            rows.append(
                {
                    "cls_name": name,
                    "conf": float(score),
                    "xyxy": [int(round(v)) for v in box.tolist()],
                }
            )
    vis = draw_expected_style(image, rows)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    imwrite_bgr(out, vis)
    extra = ROOT / "output" / "last_infer.jpg"
    imwrite_bgr(extra, vis)
    print(f"detections: {len(rows)}")
    for row in rows:
        print(f"  {row['cls_name']} {row['conf']:.3f} {row['xyxy']}")
    print(f"saved: {out}")
    print(f"saved: {extra}")


if __name__ == "__main__":
    main()
