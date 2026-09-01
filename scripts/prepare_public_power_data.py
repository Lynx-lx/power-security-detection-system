"""
下载公开电力巡检检测数据，映射为本项目 4 类 YOLO 标注。

来源（公开科研数据，非现场私有标注）：
  - CPLID（中国电力线路绝缘子）：正常绝缘子 / 缺陷
    https://github.com/InsulatorData/InsulatorDataSet
  - FOTL_Drone（输电线路异物无人机视角）：鸟巢、风筝、气球等
    https://github.com/Changping-Li/FOTL_Drone

类别映射::
  insulator          <- insulator / 绝缘子
  bird_nest          <- nest / bird_nest / 鸟巢
  foreign_object     <- kite / balloon / monkey / foreign
  damaged_insulator  <- defect / broken / 缺陷

fire / person 等无对应类的框会丢弃。

用法（项目根目录）::

    python scripts/prepare_public_power_data.py
"""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_ALIASES: dict[str, int] = {
    "insulator": 0,
    "insulators": 0,
    "normal_insulator": 0,
    "normal": 0,
    "绝缘子": 0,
    "bird_nest": 1,
    "birdnest": 1,
    "bird-nest": 1,
    "nest": 1,
    "nests": 1,
    "鸟巢": 1,
    "鸟窝": 1,
    "foreign_object": 2,
    "foreign": 2,
    "kite": 2,
    "balloon": 2,
    "monkey": 2,
    "异物": 2,
    "风筝": 2,
    "气球": 2,
    "damaged_insulator": 3,
    "defect": 3,
    "defective": 3,
    "defective_insulator": 3,
    "insulator_defect": 3,
    "broken": 3,
    "damage": 3,
    "fault": 3,
    "缺陷": 3,
    "破损": 3,
    "自爆": 3,
}

SKIP_NAMES = {
    "fire",
    "person",
    "worker",
    "people",
    "flame",
    "smoke",
    "火",
    "人",
}

# FOTL_Drone 论文类别顺序（无 classes.txt 时使用）
FOTL_INDEX_MAP = {
    0: 1,  # nest
    1: 2,  # kite
    2: 2,  # balloon
    3: None,  # fire
    4: None,  # person
    5: 2,  # monkey
}

SOURCES = (
    (
        "cplid",
        "https://github.com/InsulatorData/InsulatorDataSet.git",
    ),
    (
        "fotl",
        "https://github.com/Changping-Li/FOTL_Drone.git",
    ),
)


def _norm_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def map_class(name: str) -> int | None:
    key = _norm_name(name)
    if key in SKIP_NAMES or name.strip() in SKIP_NAMES:
        return None
    if key in CLASS_ALIASES:
        return CLASS_ALIASES[key]
    for alias, cid in CLASS_ALIASES.items():
        if alias in key or key in alias:
            return cid
    return None


def git_clone(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_dir() and any(dest.iterdir()):
        print(f"[info] reuse existing clone {dest.as_posix()}")
        return True
    print(f"[info] git clone --depth 1 {url}")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            check=True,
            cwd=str(ROOT),
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"[warn] clone failed: {type(exc).__name__}")
        return False


def voc_xml_to_yolo(xml_path: Path) -> list[str]:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return []
    size = root.find("size")
    if size is None:
        return []
    w = float(size.findtext("width") or 0)
    h = float(size.findtext("height") or 0)
    if w <= 1 or h <= 1:
        return []
    lines: list[str] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        cid = map_class(name)
        if cid is None:
            continue
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = float(box.findtext("xmin") or 0)
        ymin = float(box.findtext("ymin") or 0)
        xmax = float(box.findtext("xmax") or 0)
        ymax = float(box.findtext("ymax") or 0)
        xc = ((xmin + xmax) / 2.0) / w
        yc = ((ymin + ymax) / 2.0) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        if bw <= 0 or bh <= 0:
            continue
        xc = min(max(xc, 0.0), 1.0)
        yc = min(max(yc, 0.0), 1.0)
        bw = min(max(bw, 1e-6), 1.0)
        bh = min(max(bh, 1e-6), 1.0)
        lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


def remap_yolo_txt(txt_path: Path, names: list[str] | None, index_map: dict[int, int | None] | None) -> list[str]:
    lines_out: list[str] = []
    for raw in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        try:
            src_id = int(float(parts[0]))
            xc, yc, bw, bh = (float(x) for x in parts[1:5])
        except ValueError:
            continue
        cid: int | None = None
        if names and 0 <= src_id < len(names):
            cid = map_class(names[src_id])
        elif index_map is not None:
            cid = index_map.get(src_id)
        if cid is None:
            continue
        lines_out.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines_out


def index_images(src: Path) -> dict[str, Path]:
    table: dict[str, Path] = {}
    for p in src.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS:
            table[p.stem] = p
    return table


def convert_tree(src: Path, img_out: Path, lab_out: Path, prefix: str) -> int:
    xmls = list(src.rglob("*.xml"))
    txts = [p for p in src.rglob("*.txt") if p.name.lower() not in {"classes.txt", "readme.txt"}]
    img_index = index_images(src)
    print(
        f"[info] {prefix}: indexed_images={len(img_index)} xml={len(xmls)} txt={len(txts)}"
    )
    n = 0
    used_stems: set[str] = set()
    if xmls:
        grouped: dict[str, list[Path]] = {}
        for xml_path in xmls:
            grouped.setdefault(xml_path.stem, []).append(xml_path)
        for stem0, group in grouped.items():
            lines: list[str] = []
            for xml_path in group:
                lines.extend(voc_xml_to_yolo(xml_path))
            if not lines:
                continue
            img = img_index.get(stem0)
            if img is None:
                continue
            stem = f"{prefix}_{stem0}"
            shutil.copy2(img, img_out / f"{stem}{img.suffix.lower()}")
            (lab_out / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            used_stems.add(stem0)
            n += 1
    names = None
    names_file = src / "classes.txt"
    if names_file.is_file():
        names = [ln.strip() for ln in names_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    index_map = FOTL_INDEX_MAP if prefix == "fotl" else None
    skipped_no_img = 0
    skipped_empty = 0
    for txt_path in txts:
        if txt_path.stem in used_stems:
            continue
        lines = remap_yolo_txt(txt_path, names, index_map)
        if not lines:
            skipped_empty += 1
            continue
        img = img_index.get(txt_path.stem)
        if img is None:
            skipped_no_img += 1
            continue
        stem = f"{prefix}_{txt_path.stem}"
        shutil.copy2(img, img_out / f"{stem}{img.suffix.lower()}")
        (lab_out / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        n += 1
    if txts:
        print(f"[info] {prefix}: yolo_txt skipped empty={skipped_empty} missing_image={skipped_no_img}")
    return n


def write_splits(img_dir: Path, split_dir: Path, val_ratio: float, seed: int) -> None:
    files = sorted(p.name for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    rng = random.Random(seed)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * val_ratio)) if files else 0
    val = files[:n_val]
    train = files[n_val:]
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "train.txt").write_text("\n".join(train) + ("\n" if train else ""), encoding="utf-8")
    (split_dir / "val.txt").write_text("\n".join(val) + ("\n" if val else ""), encoding="utf-8")
    print(f"[info] split train={len(train)} val={len(val)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare public power-line detection data as 4-class YOLO")
    p.add_argument("--raw", type=str, default="data/raw")
    p.add_argument("--out", type=str, default="data/public_yolo")
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = ROOT / args.raw
    out_root = ROOT / args.out
    img_out = out_root / "images"
    lab_out = out_root / "labels"
    if img_out.exists():
        shutil.rmtree(img_out)
    if lab_out.exists():
        shutil.rmtree(lab_out)
    img_out.mkdir(parents=True)
    lab_out.mkdir(parents=True)

    if not args.skip_download:
        for name, url in SOURCES:
            git_clone(url, raw_root / name)

    total = 0
    for name, _ in SOURCES:
        src = raw_root / name
        if not src.is_dir():
            print(f"[warn] missing {src.as_posix()}")
            continue
        n = convert_tree(src, img_out, lab_out, name)
        print(f"[info] converted {n} images from {name}")
        total += n

    if total == 0:
        raise SystemExit("no labeled images converted; check clones under data/raw/")
    counts = [0, 0, 0, 0]
    for lab in lab_out.glob("*.txt"):
        for line in lab.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                cid = int(float(parts[0]))
            except ValueError:
                continue
            if 0 <= cid < 4:
                counts[cid] += 1
    names = ("insulator", "bird_nest", "foreign_object", "damaged_insulator")
    print("[info] box counts: " + ", ".join(f"{n}={c}" for n, c in zip(names, counts)))
    write_splits(img_out, out_root / "splits", args.val_ratio, args.seed)
    print(f"[info] wrote YOLO set under {args.out} ({total} images)")


if __name__ == "__main__":
    main()
