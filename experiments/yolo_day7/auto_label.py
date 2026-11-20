"""Tự gán nhãn ảnh mới trong image/ và xuất COCO cho CVAT.

Theo dõi liên tục: .venv/bin/python experiments/yolo_day7/auto_label.py
Chạy một lượt: .venv/bin/python experiments/yolo_day7/auto_label.py --once
Nhấn Ctrl+C để dừng theo dõi.
"""

import argparse
import json
import os
import tempfile
import time
import zipfile
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = ROOT / "image"
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "yolo_day7"
MODEL_PATH = ROOT / "yolo11n.pt"
GENERATOR = "pipeline.auto_label.v2"

CONFIDENCE = 0.35
IMAGE_SIZE = 640
MIN_BOX_HEIGHT = 20
DEVICE = "cpu"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

CLASSES = [
    "person", "bicycle", "motorbike", "car", "bus", "truck",
    "three_wheeler", "ignore_region",
]
CATEGORY_IDS = {name: index for index, name in enumerate(CLASSES, start=1)}
CLASS_MAP = {
    "person": "person",
    "bicycle": "bicycle",
    "motorcycle": "motorbike",
    "car": "car",
    "bus": "bus",
    "truck": "truck",
}

def signature(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def predict_image(model, path):
    with Image.open(path) as img:
        width, height = img.size
        img.verify()

    result = model.predict(
        source=str(path),
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        verbose=False,
    )[0]
    boxes = []
    for box in result.boxes:
        class_name = result.names[int(box.cls[0])]
        if class_name not in CLASS_MAP:
            continue

        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
        y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
        box_width, box_height = x2 - x1, y2 - y1
        if box_width <= 0 or box_height < MIN_BOX_HEIGHT:
            continue

        boxes.append({
            "category_id": CATEGORY_IDS[CLASS_MAP[class_name]],
            "bbox": [round(x1, 2), round(y1, 2), round(box_width, 2), round(box_height, 2)],
            "area": round(box_width * box_height, 2),
            "iscrowd": 0,
            "segmentation": [],
        })
    return width, height, boxes


def make_coco(records, input_dir):
    images, annotations = [], []
    for image_id, name in enumerate(sorted(records), start=1):
        _, width, height, boxes = records[name]
        images.append({
            "id": image_id,
            "file_name": name,
            "width": width,
            "height": height,
        })
        for box in boxes:
            annotations.append({
                "id": len(annotations) + 1,
                "image_id": image_id,
                **box,
            })
    return {
        "info": {
            "description": "YOLO11 experimental pre-labels - Day 8 taxonomy, not ground truth",
            "version": "experimental",
            "generator": GENERATOR,
            "source_dir": str(input_dir),
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": CATEGORY_IDS[name], "name": name, "supercategory": ""}
            for name in CLASSES
        ],
    }


def backup_legacy_outputs(json_path, zip_path, input_dir):
    if not json_path.exists() and not zip_path.exists():
        return
    try:
        info = json.loads(json_path.read_text(encoding="utf-8")).get("info", {})
        if info.get("generator") == GENERATOR and info.get("source_dir") == str(input_dir):
            return
    except (OSError, ValueError, TypeError, AttributeError):
        pass

    for path in (json_path, zip_path):
        if not path.exists():
            continue
        backup = path.with_name(f"{path.stem}.legacy{path.suffix}")
        index = 2
        while backup.exists():
            backup = path.with_name(f"{path.stem}.legacy_{index}{path.suffix}")
            index += 1
        path.rename(backup)
        print(f"Đã lưu bản cũ: {backup}", flush=True)


def write_outputs(coco, output_dir, input_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "auto_coco.json"
    zip_path = output_dir / "auto_coco_cvat.zip"
    json_bytes = (json.dumps(coco, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

    json_fd, temp_json = tempfile.mkstemp(prefix=".auto_coco_", dir=output_dir)
    zip_fd, temp_zip = tempfile.mkstemp(prefix=".auto_coco_cvat_", dir=output_dir)
    os.close(zip_fd)
    try:
        with os.fdopen(json_fd, "wb") as file:
            file.write(json_bytes)
        with zipfile.ZipFile(temp_zip, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("annotations/instances_default.json", json_bytes)
        backup_legacy_outputs(json_path, zip_path, input_dir)
        os.replace(temp_json, json_path)
        os.replace(temp_zip, zip_path)
    finally:
        for temp_path in (temp_json, temp_zip):
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    print(
        f"Đã cập nhật: {len(coco['images'])} ảnh, "
        f"{len(coco['annotations'])} box -> {json_path}, {zip_path}",
        flush=True,
    )


def scan(input_dir, records, observed, failed, model, once):
    paths = {
        path.name: path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    changed = False
    for name in set(records) - set(paths):
        del records[name]
        changed = True
        print(f"Đã xóa khỏi nhãn: {name}", flush=True)

    for name in sorted(paths):
        path = paths[name]
        try:
            current = signature(path)
        except OSError:
            continue
        if not once and observed.get(name) != current:
            observed[name] = current
            continue  # Chờ một nhịp để ảnh đang được chép xong.
        if name in records and records[name][0] == current:
            continue
        if failed.get(name) == current:
            continue
        try:
            width, height, boxes = predict_image(model, path)
            if signature(path) != current:
                continue  # Ảnh đổi trong lúc suy luận; thử lại ở vòng sau.
        except (OSError, ValueError, RuntimeError) as exc:
            failed[name] = current
            if name in records:
                del records[name]
                changed = True
            print(f"Bỏ qua {name}: {exc}", flush=True)
            continue
        records[name] = (current, width, height, boxes)
        failed.pop(name, None)
        changed = True
        print(f"Đã gán nhãn {name}: {len(boxes)} box", flush=True)

    for name in set(observed) - set(paths):
        del observed[name]
    for name in set(failed) - set(paths):
        del failed[name]
    return changed


def main():
    parser = argparse.ArgumentParser(description="Theo dõi image/ và tự gán nhãn COCO cho ảnh mới")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--once", action="store_true", help="Gán nhãn một lượt rồi thoát")
    parser.add_argument("--interval", type=float, default=2.0, help="Số giây giữa các lần quét")
    args = parser.parse_args()
    input_dir = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    if not input_dir.is_dir():
        parser.error(f"Không tìm thấy thư mục ảnh: {input_dir}")
    if not MODEL_PATH.is_file():
        parser.error(f"Không tìm thấy model: {MODEL_PATH}")
    if args.interval <= 0:
        parser.error("--interval phải lớn hơn 0")

    print(f"Đang đọc ảnh từ: {input_dir}", flush=True)
    if not args.once:
        print("Đang theo dõi ảnh mới; nhấn Ctrl+C để dừng.", flush=True)

    model = None
    records, observed, failed = {}, {}, {}
    try:
        while True:
            candidates = any(
                path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                for path in input_dir.iterdir()
            )
            if candidates and model is None:
                print(f"Đang tải model: {MODEL_PATH}", flush=True)
                model = YOLO(str(MODEL_PATH))
            if model is not None and scan(input_dir, records, observed, failed, model, args.once):
                write_outputs(make_coco(records, input_dir), output_dir, input_dir)
            if args.once:
                if not candidates:
                    print("Không có ảnh mới để gán nhãn.", flush=True)
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Đã dừng theo dõi.", flush=True)


if __name__ == "__main__":
    main()
