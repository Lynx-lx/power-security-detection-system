"""本地 Gradio 演示：上传图片 → YOLOv5sLite 检测 → 风险评估。

启动（项目根目录）::

    pip install -r requirements.txt
    python app.py

加载优先级：yolov5s_lite.pt（业务微调）→ yolov5s_lite_demo.pt（COCO 主干、头随机）→ 模拟框。
不把本机绝对路径返回给前端。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from viscale.detection import POWER_SECURITY_CLASSES, build_yolov5s_lite
from viscale.detection.decode import nms as boxes_nms
from viscale.io.camera import load_camera_config
from viscale.measurement.scale import meters_per_pixel_at_depth
from viscale.risk import DetectionRecord, RiskAssessor

# 权重只放在仓库相对路径下；默认文件名，不把 .pt 打进代码仓库
CHECKPOINTS_REL = "models/checkpoints"
REAL_WEIGHTS_REL = f"{CHECKPOINTS_REL}/yolov5s_lite.pt"
DEMO_WEIGHTS_REL = f"{CHECKPOINTS_REL}/yolov5s_lite_demo.pt"
WORLD_WEIGHTS_REL = f"{CHECKPOINTS_REL}/yolov8s-world.pt"
POWER_ULTRA_REL = f"{CHECKPOINTS_REL}/yolov8n_power.pt"
DEFAULT_WEIGHTS_REL = REAL_WEIGHTS_REL
DEFAULT_WEIGHTS = os.environ.get("VISCALE_WEIGHTS", "")
DEFAULT_ATTN = os.environ.get("VISCALE_ATTN", "eca")
DEFAULT_DEVICE = os.environ.get("VISCALE_DEVICE", "cpu")
DEFAULT_IMGSZ = int(os.environ.get("VISCALE_IMGSZ", "640"))

DEMO_MODE_HINT = (
    "⚠️【演示模拟模式】未找到预训练权重，展示模拟检测框，放置权重文件可获得真实检测效果"
)
WEIGHTS_LOAD_FAIL_HINT = (
    "⚠️【演示模拟模式】预训练权重读取失败，展示模拟检测框，放置权重文件可获得真实检测效果"
)
WEIGHTS_OK_HINT = "已加载微调权重 yolov5s_lite.pt（models/checkpoints/）。"
WEIGHTS_ADAPTER_HINT = (
    "⚠️已加载 yolov5s_lite_demo.pt：仅 COCO 主干部分迁移、检测头随机初始化，"
    "未做电力数据微调，不具备真实电力目标识别能力。"
)
WEIGHTS_WORLD_HINT = (
    "已用 YOLO-World 开放词汇检测（绝缘子/鸟巢/异物文本提示）。"
    "自研 yolov5s_lite 短训权重对生成图几乎无效，故默认走该后端。"
)
WEIGHTS_ULTRA_HINT = "已加载公开 CPLID+FOTL 微调权重 yolov8n_power.pt。"
OUTPUT_REL = "output"

BOX_COLORS = [
    (46, 204, 113),
    (52, 152, 219),
    (241, 196, 15),
    (230, 126, 34),
    (155, 89, 182),
    (231, 76, 60),
    (26, 188, 156),
    (149, 165, 166),
]
GRADE_COLOR = {1: "#27ae60", 2: "#2980b9", 3: "#e67e22", 4: "#c0392b"}

_lock = threading.Lock()
_runtime: dict = {
    "model": None,
    "device": None,
    "assessor": None,
    "weights_note": "",
    "camera_note": "",
    "mpp_from_camera": None,
    "demo_mode": True,
    "weights_loaded": False,
    "weights_arg": REAL_WEIGHTS_REL,
    "infer_kind": "mock",
    "openvocab": os.environ.get("VISCALE_OPENVOCAB", "0") == "1",
}


def _resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")


def _redact_abs_paths(text: str) -> str:
    """前端展示用：去掉 Windows/Unix 绝对路径，避免 C:/Users/... 出现在页面上。"""
    if not text:
        return ""
    text = re.sub(r"[A-Za-z]:[\\/][^\s<>\"']+", "[local]", text)
    text = re.sub(r"(?<![A-Za-z:])/(?:Users|home|mnt)/[^\s<>\"']+", "[local]", text)
    return text


def _weights_path(path: str) -> Path:
    p = Path(path) if path else Path(DEFAULT_WEIGHTS_REL)
    if not p.is_absolute():
        p = ROOT / p
    return p


def _checkpoint_exists(path: str | None = None) -> bool:
    p = _weights_path(path or DEFAULT_WEIGHTS_REL)
    try:
        return p.is_file() and p.stat().st_size > 0
    except OSError:
        return False


def _select_weight_file(preferred: str | None) -> tuple[str | None, str]:
    """优先真实微调权重，其次 adapter demo 权重，否则无文件。kind: real|coco_demo|mock"""
    cand: list[tuple[str, str]] = []
    if preferred and str(preferred).strip():
        cand.append((preferred.strip(), "custom"))
    cand.append((REAL_WEIGHTS_REL, "real"))
    cand.append((DEMO_WEIGHTS_REL, "coco_demo"))
    seen: set[str] = set()
    for rel, kind in cand:
        if rel in seen:
            continue
        seen.add(rel)
        if not _checkpoint_exists(rel):
            continue
        if kind == "custom":
            kind = "coco_demo" if "demo" in Path(rel).name.lower() else "real"
        return rel, kind
    return None, "mock"


def _torch_load(path: Path, device: torch.device):
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


def _unwrap_ckpt_state(ckpt):
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    return state


def _try_load_weights(
    model: torch.nn.Module,
    path: str | None,
    device: torch.device,
) -> tuple[bool, str, str]:
    """返回 (是否加载了可前向的权重, 前端提示, infer_kind)。"""
    rel, kind = _select_weight_file(path)
    if rel is None or kind == "mock":
        print("[info] no yolov5s_lite.pt or yolov5s_lite_demo.pt; mock boxes")
        return False, DEMO_MODE_HINT, "mock"
    try:
        ckpt = _torch_load(_weights_path(rel), device)
        state = _unwrap_ckpt_state(ckpt)
        if not isinstance(state, dict):
            print("[warn] checkpoint format unsupported; mock boxes")
            return False, WEIGHTS_LOAD_FAIL_HINT, "mock"
        model.load_state_dict(state, strict=False)
        if kind == "coco_demo":
            print("[warn] loaded yolov5s_lite_demo.pt: COCO backbone only, random detect head, NOT power-trained")
            return True, WEIGHTS_ADAPTER_HINT, "coco_demo"
        print("[info] loaded yolov5s_lite.pt from models/checkpoints/")
        return True, WEIGHTS_OK_HINT, "real"
    except (OSError, FileNotFoundError, KeyError, RuntimeError, ValueError, TypeError):
        print("[warn] checkpoint load failed; mock boxes")
        return False, WEIGHTS_LOAD_FAIL_HINT, "mock"
    except Exception:
        print("[warn] checkpoint load failed; mock boxes")
        return False, WEIGHTS_LOAD_FAIL_HINT, "mock"


def init_runtime(
    weights: str,
    attn: str,
    device_name: str,
    camera_config: str | None = None,
    working_distance_m: float = 0.0,
    openvocab: bool | None = None,
) -> None:
    with _lock:
        device = _resolve_device(device_name)
        model = build_yolov5s_lite(num_classes=len(POWER_SECURITY_CLASSES), attn=attn).to(device)
        model.eval()
        loaded, note, kind = _try_load_weights(model, weights, device)
        cam_path = camera_config or "config/camera_sensor/camera.yaml"
        cam, _cam_msg = load_camera_config(cam_path)
        mpp = meters_per_pixel_at_depth(cam, working_distance_m) if working_distance_m > 0 else None
        if cam is None:
            cam_ui = "未找到本地相机标定（config/camera_sensor/camera.yaml），已跳过内参尺度。"
        else:
            cam_ui = "已读取相机配置（config/camera_sensor/）。"
            if cam.is_template:
                cam_ui += " 当前为模板占位，不可当作现场标定。"
        _runtime["model"] = model
        _runtime["device"] = device
        _runtime["assessor"] = RiskAssessor()
        _runtime["weights_note"] = _redact_abs_paths(note)
        _runtime["camera_note"] = _redact_abs_paths(cam_ui)
        _runtime["mpp_from_camera"] = mpp
        selected, _ = _select_weight_file(weights)
        _runtime["demo_mode"] = kind == "mock"
        _runtime["weights_loaded"] = loaded
        _runtime["weights_arg"] = selected or ""
        _runtime["infer_kind"] = kind
        _runtime["openvocab"] = (
            bool(openvocab) if openvocab is not None else os.environ.get("VISCALE_OPENVOCAB", "0") == "1"
        )
        _runtime["ultra_model"] = None
        ultra_path = ROOT / POWER_ULTRA_REL
        if (not _runtime["openvocab"]) and ultra_path.is_file():
            try:
                from ultralytics import YOLO

                _runtime["ultra_model"] = YOLO(str(ultra_path))
                _runtime["infer_kind"] = "ultra"
                _runtime["demo_mode"] = False
                _runtime["weights_loaded"] = True
                _runtime["weights_note"] = WEIGHTS_ULTRA_HINT
                print("[info] loaded", POWER_ULTRA_REL)
            except Exception as exc:
                print("[warn] yolov8n_power load failed:", type(exc).__name__)
        print(
            "[info] infer_kind=%s openvocab=%s demo_mode=%s device=%s"
            % (_runtime["infer_kind"], _runtime["openvocab"], _runtime["demo_mode"], device)
        )


def letterbox(image_bgr: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    h, w = image_bgr.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, scale, (left, top)


def to_tensor(image_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = image_bgr[:, :, ::-1]
    x = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return torch.from_numpy(x).unsqueeze(0).to(device)


def map_xyxy(xyxy: list[float], pad: tuple[int, int], scale: float, wh: tuple[int, int]) -> tuple[int, int, int, int]:
    left, top = pad
    w, h = wh
    x1 = int(np.clip((xyxy[0] - left) / scale, 0, w - 1))
    y1 = int(np.clip((xyxy[1] - top) / scale, 0, h - 1))
    x2 = int(np.clip((xyxy[2] - left) / scale, 0, w - 1))
    y2 = int(np.clip((xyxy[3] - top) / scale, 0, h - 1))
    return x1, y1, x2, y2


def draw_boxes(image_rgb: np.ndarray, rows: list[dict]) -> np.ndarray:
    vis = image_rgb.copy()
    bgr = vis[:, :, ::-1].copy()
    for row in rows:
        x1, y1, x2, y2 = row["xyxy"]
        cid = int(row["cls_id"])
        color = BOX_COLORS[cid % len(BOX_COLORS)]
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        label = f"{row['cls_name']} {row['conf']:.2f}"
        cv2.putText(bgr, label, (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return bgr[:, :, ::-1]


def _mock_detections(height: int, width: int) -> list[dict]:
    """演示模拟模式：按图像宽高构造 2～4 个假框（非真实检测）。禁止返回空列表。"""
    h = max(int(height), 32)
    w = max(int(width), 32)
    names = list(POWER_SECURITY_CLASSES)
    name_to_id = {n: i for i, n in enumerate(names)}
    presets = (
        (0.08, 0.18, 0.36, 0.62, "insulator", 0.72),
        (0.42, 0.12, 0.68, 0.40, "bird_nest", 0.61),
        (0.58, 0.48, 0.88, 0.82, "foreign_object", 0.55),
        (0.18, 0.58, 0.40, 0.86, "damaged_insulator", 0.48),
    )
    rows: list[dict] = []
    for x1r, y1r, x2r, y2r, name, score in presets:
        x1 = int(np.clip(round(x1r * w), 0, w - 2))
        y1 = int(np.clip(round(y1r * h), 0, h - 2))
        x2 = int(np.clip(round(x2r * w), x1 + 8, w - 1))
        y2 = int(np.clip(round(y2r * h), y1 + 8, h - 1))
        cid = int(name_to_id.get(name, 0))
        rows.append(
            {
                "cls_id": cid,
                "cls_name": name,
                "conf": float(score),
                "xyxy": [int(x1), int(y1), int(x2), int(y2)],
            }
        )
    return rows[:4]


def _filter_demo_boxes(rows: list[dict], conf_thres: float, iou_thres: float) -> list[dict]:
    """对模拟框走与真实推理相同的置信度过滤 + NMS；过滤后若为空则回退至少 2 框。"""
    kept = [r for r in rows if float(r["conf"]) >= float(conf_thres)]
    if len(kept) >= 2:
        boxes = torch.tensor([r["xyxy"] for r in kept], dtype=torch.float32)
        scores = torch.tensor([r["conf"] for r in kept], dtype=torch.float32)
        order = boxes_nms(boxes, scores, float(iou_thres))
        kept = [kept[int(i)] for i in order.tolist()]
    if len(kept) < 2:
        # 演示模式禁止空列表：保留置信度最高的 2 个原始模拟框
        kept = sorted(rows, key=lambda r: r["conf"], reverse=True)[:2]
    return kept


def _save_vis_relative(vis_rgb: np.ndarray) -> None:
    """可视化写入仓库相对目录 output/，路径不返回前端。"""
    try:
        out_dir = ROOT / OUTPUT_REL
        out_dir.mkdir(parents=True, exist_ok=True)
        bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / "last_infer.jpg"), bgr)
    except OSError:
        print("[warn] could not write visualization under output/")


def as_rgb(image) -> np.ndarray | None:
    """接受 Gradio numpy / filepath / ImageData / PIL。返回 RGB uint8。"""
    if image is None:
        return None
    if isinstance(image, dict):
        image = image.get("path") or image.get("orig_path") or image.get("url")
    if hasattr(image, "path") and not isinstance(image, (str, Path, np.ndarray)):
        image = getattr(image, "path", None)
    if isinstance(image, (str, Path)):
        try:
            data = np.fromfile(str(image), dtype=np.uint8)
        except OSError:
            return None
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if hasattr(image, "convert"):
        image = np.asarray(image.convert("RGB"))
    arr = np.asarray(image)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.ndim != 3:
        return None
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    return arr


def as_bgr(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def run_infer(image, conf: float, iou: float, meters_per_pixel: float, imgsz: int):
    try:
        if _runtime.get("assessor") is None:
            init_runtime(DEFAULT_WEIGHTS, DEFAULT_ATTN, DEFAULT_DEVICE)

        rgb = as_rgb(image)
        if rgb is None:
            return None, "请先上传本地图片（可从 data/ 选择测试图）。", {"headers": ["类别", "置信度", "x1", "y1", "x2", "y2"], "data": []}

        assessor: RiskAssessor = _runtime["assessor"]
        names = list(POWER_SECURITY_CLASSES)
        model = _runtime.get("model")
        if model is not None and getattr(model, "class_names", None):
            names = list(model.class_names)
        h, w = int(rgb.shape[0]), int(rgb.shape[1])

        kind = _runtime.get("infer_kind") or "mock"
        weight_rel = _runtime.get("weights_arg") or ""
        use_openvocab = bool(_runtime.get("openvocab"))
        if use_openvocab:
            from viscale.detection.open_vocab import WORLD_WEIGHT_REL, ensure_world_weights, predict_open_vocab

            image_bgr = as_bgr(rgb)
            wpath = ROOT / WORLD_WEIGHT_REL
            try:
                ensure_world_weights(wpath)
                rows = predict_open_vocab(
                    image_bgr,
                    wpath,
                    conf=float(conf),
                    iou=float(iou),
                    device=str(_runtime.get("device") or "cpu"),
                )
                _runtime["weights_note"] = WEIGHTS_WORLD_HINT
                _runtime["infer_kind"] = "openvocab"
            except Exception as exc:
                print("[warn] open-vocab infer failed:", type(exc).__name__)
                use_openvocab = False
                rows = []
        else:
            rows = []

        use_weights = (
            (not use_openvocab)
            and bool(_runtime.get("weights_loaded"))
            and kind in ("real", "coco_demo")
            and bool(weight_rel)
            and _checkpoint_exists(weight_rel)
        )

        ultra_model = _runtime.get("ultra_model")
        if use_openvocab:
            pass
        elif (not use_openvocab) and ultra_model is not None:
            image_bgr = as_bgr(rgb)
            result = ultra_model.predict(
                image_bgr,
                conf=float(conf),
                iou=float(iou),
                imgsz=int(imgsz),
                device=str(_runtime.get("device") or "cpu"),
                verbose=False,
            )[0]
            rows = []
            names = list(POWER_SECURITY_CLASSES)
            if result.boxes is not None:
                for box, score, cid in zip(
                    result.boxes.xyxy.cpu().numpy(),
                    result.boxes.conf.cpu().numpy(),
                    result.boxes.cls.cpu().numpy().astype(int),
                ):
                    cid_i = int(cid)
                    name = names[cid_i] if 0 <= cid_i < len(names) else str(cid_i)
                    rows.append(
                        {
                            "cls_id": cid_i,
                            "cls_name": name,
                            "conf": float(score),
                            "xyxy": [int(round(v)) for v in box.tolist()],
                        }
                    )
            _runtime["weights_note"] = WEIGHTS_ULTRA_HINT
            _runtime["infer_kind"] = "ultra"
            _runtime["demo_mode"] = False
        elif use_weights and model is not None:
            image_bgr = as_bgr(rgb)
            canvas, scale, pad = letterbox(image_bgr, int(imgsz))
            tensor = to_tensor(canvas, _runtime["device"])
            with torch.inference_mode():
                dets = model.predict(tensor, conf_thres=float(conf), iou_thres=float(iou))[0]
            rows = []
            for *xyxy, score, cls_id in dets.cpu().tolist():
                mapped = map_xyxy(xyxy, pad, scale, (w, h))
                cid = int(cls_id)
                name = names[cid] if 0 <= cid < len(names) else str(cid)
                rows.append(
                    {
                        "cls_id": cid,
                        "cls_name": name,
                        "conf": float(score),
                        "xyxy": [int(mapped[0]), int(mapped[1]), int(mapped[2]), int(mapped[3])],
                    }
                )
        else:
            print("[info] demo simulation: building mock boxes (no checkpoint)")
            _runtime["demo_mode"] = True
            _runtime["weights_note"] = DEMO_MODE_HINT
            rows = _filter_demo_boxes(_mock_detections(h, w), float(conf), float(iou))
            if len(rows) < 2:
                rows = _mock_detections(h, w)[:3]

        records = [
            DetectionRecord(
                cls_name=r["cls_name"],
                conf=float(r["conf"]),
                xyxy=tuple(float(v) for v in r["xyxy"]),
            )
            for r in rows
        ]
        mpp = float(meters_per_pixel) if meters_per_pixel and meters_per_pixel > 0 else _runtime.get("mpp_from_camera")
        report = assessor.assess(records, meters_per_pixel=mpp, image_wh=(w, h), update_elc=False)

        vis = draw_boxes(rgb, rows)
        _save_vis_relative(vis)
        color = GRADE_COLOR.get(int(report.grade), "#333")
        viol_lines = "无"
        if report.violations:
            viol_lines = "\n".join(
                f"- **{ev.kind}** 分数 {ev.score:.2f}：{ev.message}" for ev in report.violations
            )
        cam_line = _redact_abs_paths(_runtime.get("camera_note") or "")
        weights_line = _redact_abs_paths(_runtime.get("weights_note") or "")
        md = (
            f"<div style='padding:12px;border-radius:8px;border:1px solid {color};'>"
            f"<p style='margin:0;color:{color};font-size:22px;font-weight:700;'>风险等级：{report.label_zh}</p>"
            f"<p style='margin:8px 0 0;'>综合分数 <b>{report.score:.3f}</b>"
            f" ｜ 等级 {int(report.grade)} / 4"
            f" ｜ ELC 阈值 {[round(t, 3) for t in report.thresholds]}</p>"
            f"<p style='margin:8px 0 0;color:#666;font-size:13px;'>{weights_line}</p>"
            f"<p style='margin:8px 0 0;color:#666;font-size:13px;'>{cam_line}</p>"
            f"</div>\n\n**违规项**\n\n{viol_lines}\n\n"
            f"**检测框数量：** {len(rows)}"
        )
        table = {
            "headers": ["类别", "置信度", "x1", "y1", "x2", "y2"],
            "data": [
                [r["cls_name"], f"{r['conf']:.3f}", r["xyxy"][0], r["xyxy"][1], r["xyxy"][2], r["xyxy"][3]]
                for r in rows
            ],
        }
        return vis, md, table
    except Exception as exc:
        print("[warn] infer pipeline failed:", type(exc).__name__)
        return None, "推理失败，请检查上传图片后重试。", {"headers": ["类别", "置信度", "x1", "y1", "x2", "y2"], "data": []}


def build_ui(imgsz: int):
    import gradio as gr

    with gr.Blocks(title="电力安防 · 检测与风险评估") as demo:
        gr.Markdown(
            "## 电力安防视觉检测演示\n"
            "上传本地图片，检测绝缘子 / 鸟巢 / 异物 / 破损绝缘子，并输出风险等级。\n"
            "本地若有 `models/checkpoints/yolov8n_power.pt`，优先用公开数据微调权重。"
        )
        with gr.Row():
            inp = gr.Image(type="filepath", label="上传图片", sources=["upload"])
            out = gr.Image(type="numpy", label="检测框可视化")
        with gr.Row():
            conf = gr.Slider(0.01, 0.90, value=0.25, step=0.01, label="置信度阈值")
            iou = gr.Slider(0.10, 0.90, value=0.45, step=0.05, label="NMS IoU")
            mpp = gr.Number(value=0.0, label="米/像素（0 表示不启用尺度）")
        btn = gr.Button("检测并评估风险", variant="primary")
        risk = gr.Markdown()
        table = gr.Dataframe(
            headers=["类别", "置信度", "x1", "y1", "x2", "y2"],
            label="检测列表",
            interactive=False,
        )
        btn.click(
            fn=lambda im, c, u, m: run_infer(im, c, u, m, imgsz),
            inputs=[inp, conf, iou, mpp],
            outputs=[out, risk, table],
            api_name="infer",
        )
        example_paths = [
            str(ROOT / "data" / "samples" / name)
            for name in ("scene2_orig.png", "scene3_orig.png", "scene1_orig.png")
            if (ROOT / "data" / "samples" / name).is_file()
        ]
        if example_paths:
            gr.Examples(examples=[[p] for p in example_paths], inputs=inp, label="示例巡检图")
    return demo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gradio 本地检测 + 风险评估演示")
    p.add_argument(
        "--weights",
        type=str,
        default=DEFAULT_WEIGHTS,
        help="可选指定权重；默认自动：yolov5s_lite.pt → yolov5s_lite_demo.pt → 模拟框",
    )
    p.add_argument("--attn", type=str, default=DEFAULT_ATTN, choices=("eca", "se", "cbam", "none"))
    p.add_argument(
        "--openvocab",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("VISCALE_OPENVOCAB", "0") == "1",
        help="VISCALE_OPENVOCAB=1 或 --openvocab 时用 YOLO-World；默认走 yolov5s_lite.pt",
    )
    p.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    p.add_argument(
        "--camera-config",
        type=str,
        default="config/camera_sensor/camera.yaml",
        help="本地相机 YAML（默认 config/camera_sensor/camera.yaml；不存在则提示并跳过）",
    )
    p.add_argument(
        "--working-distance-m",
        type=float,
        default=0.0,
        help="目标大致深度（米），与真实内参一起估算米/像素；0 表示不估算",
    )
    return p.parse_args()


def main() -> None:
    import gradio as gr

    args = parse_args()
    init_runtime(
        weights=args.weights,
        attn=args.attn,
        device_name=args.device,
        camera_config=args.camera_config,
        working_distance_m=args.working_distance_m,
        openvocab=args.openvocab,
    )
    demo = build_ui(args.imgsz)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=False,
        show_error=True,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
