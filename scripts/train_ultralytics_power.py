"""用 Ultralytics YOLOv8n + 公开 CPLID/FOTL 四类数据微调，导出本地权重。

比自研短训头更接近「白框+类别名」的预期可视化。权重不入库。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from viscale.detection.yolov5s_lite import POWER_SECURITY_CLASSES

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def write_image_list(data_root: Path, split_file: Path, out_list: Path) -> int:
    names = [ln.strip() for ln in split_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    img_dir = data_root / "images"
    lines: list[str] = []
    for name in names:
        stem = Path(name).stem
        hit = None
        for ext in IMAGE_EXTS:
            cand = img_dir / f"{stem}{ext}"
            if cand.is_file():
                hit = cand
                break
            cand = img_dir / name
            if cand.is_file():
                hit = cand
                break
        if hit is None:
            continue
        lines.append(hit.resolve().as_posix())
    out_list.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def write_yaml(data_root: Path) -> Path:
    n_train = write_image_list(data_root, data_root / "splits" / "train.txt", data_root / "splits" / "train_paths.txt")
    n_val = write_image_list(data_root, data_root / "splits" / "val.txt", data_root / "splits" / "val_paths.txt")
    yaml_path = data_root / "data.yaml"
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(POWER_SECURITY_CLASSES))
    yaml_path.write_text(
        (
            f"path: {data_root.resolve().as_posix()}\n"
            f"train: splits/train_paths.txt\n"
            f"val: splits/val_paths.txt\n"
            f"nc: {len(POWER_SECURITY_CLASSES)}\n"
            f"names:\n{names}\n"
        ),
        encoding="utf-8",
    )
    print(f"[info] yaml train={n_train} val={n_val} -> {yaml_path}")
    return yaml_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/public_yolo")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--model", type=str, default="yolov8n.pt")
    p.add_argument("--out", type=str, default="models/checkpoints/yolov8n_power.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = ROOT / args.data
    yaml_path = write_yaml(data_root)
    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        project=str(ROOT / "runs" / "detect"),
        name="power_yolo",
        exist_ok=True,
        pretrained=True,
        patience=8,
        plots=False,
        verbose=True,
    )
    best = ROOT / "runs" / "detect" / "power_yolo" / "weights" / "best.pt"
    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if best.is_file():
        import shutil

        shutil.copy2(best, out)
        print(f"[info] copied {best.as_posix()} -> {out.as_posix()}")
    else:
        raise SystemExit("ultralytics best.pt not found")


if __name__ == "__main__":
    main()
