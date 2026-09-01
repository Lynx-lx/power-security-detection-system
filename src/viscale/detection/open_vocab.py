"""开放词汇检测（YOLO-World）：用于公开闭集权重尚未覆盖的场景图。

本仓库自研 YOLOv5sLite 在少量 CPLID 上短训后，对「豆包生成」的整塔巡检图
几乎没有可用目标置信度，且训练时未纳入 FOTL 鸟巢/异物。开放词汇模型用
文本类别提示框出绝缘子、鸟巢、异物，类别 id 仍映射到 POWER_SECURITY_CLASSES。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from viscale.detection.yolov5s_lite import POWER_SECURITY_CLASSES

WORLD_WEIGHT_REL = "models/checkpoints/yolov8s-world.pt"
WORLD_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-world.pt"

# 与 4 类顺序一致；提示写具体一些，便于开放词汇对齐塔上目标
CLASS_PROMPTS = (
    "ceramic or glass insulator string on transmission tower",
    "bird nest on steel lattice tower",
    "white debris or foreign object hanging on power line",
    "broken or damaged insulator",
)


def ensure_world_weights(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 1_000_000:
        return path
    import urllib.request

    print("[info] downloading YOLO-World weights (ultralytics yolov8s-world.pt)")
    urllib.request.urlretrieve(WORLD_URL, str(path))
    return path


def predict_open_vocab(
    image_bgr: np.ndarray,
    weights: Path,
    conf: float = 0.15,
    iou: float = 0.45,
    device: str = "cpu",
) -> list[dict]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    if hasattr(model, "set_classes"):
        model.set_classes(list(CLASS_PROMPTS))
    h, w = image_bgr.shape[:2]
    result = model.predict(
        source=image_bgr,
        conf=float(conf),
        iou=float(iou),
        device=device,
        verbose=False,
    )[0]
    rows: list[dict] = []
    if result.boxes is None or len(result.boxes) == 0:
        return rows
    xyxy = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    clses = result.boxes.cls.cpu().numpy().astype(int)
    names = list(POWER_SECURITY_CLASSES)
    for box, score, cid in zip(xyxy, scores, clses):
        cid = int(cid)
        if cid < 0 or cid >= len(names):
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in box.tolist())
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w - 1))
        y2 = max(y1 + 1, min(y2, h - 1))
        rows.append(
            {
                "cls_id": cid,
                "cls_name": names[cid],
                "conf": float(score),
                "xyxy": [x1, y1, x2, y2],
            }
        )
    return rows
