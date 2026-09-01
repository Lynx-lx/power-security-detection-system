"""Simple inference demo for YOLOv5sLite (P2 small-object branch + attention)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from viscale.detection import POWER_SECURITY_CLASSES, build_yolov5s_lite
from viscale.io.camera import load_camera_config


def letterbox(image: np.ndarray, size: int = 640) -> tuple[np.ndarray, float, tuple[int, int]]:
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    import cv2

    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, scale, (left, top)


def to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    x = image[:, :, ::-1].transpose(2, 0, 1)
    x = np.ascontiguousarray(x, dtype=np.float32) / 255.0
    return torch.from_numpy(x).unsqueeze(0).to(device)


def make_synth_image(size: int = 640) -> np.ndarray:
    rng = np.random.default_rng(0)
    img = np.full((size, size, 3), 40, dtype=np.uint8)
    img[:, :] = (32, 48, 36)
    for _ in range(18):
        x1, y1 = int(rng.integers(20, size - 80)), int(rng.integers(20, size - 80))
        x2, y2 = x1 + int(rng.integers(12, 70)), y1 + int(rng.integers(12, 70))
        color = tuple(int(c) for c in rng.integers(80, 220, size=3))
        img[y1:y2, x1:x2] = color
    return img


def draw_dets(image: np.ndarray, dets: torch.Tensor, names: list[str], pad: tuple[int, int], scale: float) -> np.ndarray:
    import cv2

    vis = image.copy()
    left, top = pad
    for *xyxy, conf, cls in dets.cpu().tolist():
        x1 = int((xyxy[0] - left) / scale)
        y1 = int((xyxy[1] - top) / scale)
        x2 = int((xyxy[2] - left) / scale)
        y2 = int((xyxy[3] - top) / scale)
        cid = int(cls)
        label = f"{names[cid] if cid < len(names) else cid} {conf:.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 80), 2)
        cv2.putText(vis, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 80), 1)
    return vis


def imread_bgr(path: str | Path) -> np.ndarray | None:
    """Windows 中文路径下 cv2.imread 会失败，改用 fromfile + imdecode。"""
    import cv2

    p = Path(path)
    try:
        data = np.fromfile(str(p), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_bgr(path: str | Path, image: np.ndarray) -> bool:
    import cv2

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(p.suffix or ".jpg", image)
    if not ok:
        return False
    buf.tofile(str(p))
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLOv5sLite 推理 demo（小目标 P2 + 注意力）")
    p.add_argument("--image", type=str, default="", help="输入图像；为空则生成合成图")
    p.add_argument("--weights", type=str, default="", help="可选 .pt 权重")
    p.add_argument("--attn", type=str, default="eca", choices=("eca", "se", "cbam", "none"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.02)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default=str(ROOT / "outputs" / "detections" / "demo_yolov5s_lite.jpg"))
    p.add_argument("--backend", type=str, default="world", choices=("world", "lite"), help="world=YOLO-World；lite=自研短训权重")
    p.add_argument("--camera-config", type=str, default="", help="相机 YAML；默认查找 config/camera_sensor/camera.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    if args.image:
        image = imread_bgr(args.image)
        if image is None:
            raise FileNotFoundError(args.image)
    else:
        image = make_synth_image(args.imgsz)
        print("using synthetic image")

    if args.backend == "world":
        from viscale.detection.open_vocab import WORLD_WEIGHT_REL, ensure_world_weights, predict_open_vocab

        wpath = ensure_world_weights(ROOT / WORLD_WEIGHT_REL)
        rows = predict_open_vocab(image, wpath, conf=args.conf, iou=args.iou, device=str(device))
        vis = image.copy()
        import cv2

        for row in rows:
            x1, y1, x2, y2 = row["xyxy"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 80), 2)
            label = f"{row['cls_name']} {row['conf']:.2f}"
            cv2.putText(vis, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 80), 1)
        print(f"open-vocab detections: {len(rows)}")
        for row in rows:
            print(f"  {row['cls_name']} {row['conf']:.3f} {row['xyxy']}")
        out = Path(args.out)
        if not imwrite_bgr(out, vis):
            raise OSError(f"failed to write {out}")
        extra = ROOT / "output" / "last_infer.jpg"
        imwrite_bgr(extra, vis)
        print(f"saved: {out}")
        print(f"saved: {extra}")
        return

    model = build_yolov5s_lite(num_classes=len(POWER_SECURITY_CLASSES), attn=args.attn).to(device)
    if args.weights:
        try:
            ckpt = torch.load(args.weights, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(args.weights, map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        print(f"loaded weights: {args.weights}")
    else:
        print("no weights given; using random init (boxes will be noisy)")

    print(f"attn={args.attn} params={model.parameter_count() / 1e6:.2f}M device={device}")
    _cam, cam_msg = load_camera_config(args.camera_config or None)
    print(cam_msg)

    canvas, scale, pad = letterbox(image, args.imgsz)
    tensor = to_tensor(canvas, device)

    model.eval()
    with torch.inference_mode():
        raw = model(tensor)
        print(f"decoded pred shape: {tuple(raw.shape)} (4 heads: P2/4, P3/8, P4/16, P5/32)")
        dets = model.predict(tensor, conf_thres=args.conf, iou_thres=args.iou)[0]

    print(f"detections: {dets.shape[0]}")
    vis = draw_dets(image, dets, model.class_names, pad, scale)
    out = Path(args.out)
    if not imwrite_bgr(out, vis):
        raise OSError(f"failed to write {out}")
    extra = ROOT / "output" / "last_infer.jpg"
    imwrite_bgr(extra, vis)
    print(f"saved: {out}")
    print(f"saved: {extra}")


if __name__ == "__main__":
    main()
