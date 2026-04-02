import xml.etree.ElementTree as ET
from pathlib import Path

import cv2


SUPPORTED_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff")


def _read_image(img_path, grayscale=True):
    read_mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    return cv2.imread(str(img_path), read_mode)


def _postprocess_image(img, size=256, enhance_contrast=False, denoise=False):
    if img is None:
        return None

    if denoise:
        # Median blur keeps defect edges better than heavy Gaussian filtering.
        img = cv2.medianBlur(img, 3)

    if enhance_contrast:
        if len(img.shape) == 2:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)
        else:
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            y_channel, cr, cb = cv2.split(ycrcb)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            y_channel = clahe.apply(y_channel)
            ycrcb = cv2.merge((y_channel, cr, cb))
            img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    return cv2.resize(img, (size, size))


def load_and_preprocess_dataset(
    raw_dir,
    processed_dir,
    size=256,
    grayscale=True,
    enhance_contrast=False,
    denoise=False,
):
    """
    Standard preprocessing for folder-structured datasets:
    raw_dir/class_name/*.jpg -> processed_dir/class_name/*.jpg
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Original data directory not found: {raw_dir}")

    total_images = 0
    for class_dir in raw_dir.iterdir():
        if not class_dir.is_dir():
            continue

        out_class_dir = processed_dir / class_dir.name
        out_class_dir.mkdir(exist_ok=True)
        print(f"Processing class: {class_dir.name}")
        class_count = 0

        for ext in SUPPORTED_EXTENSIONS:
            for img_path in class_dir.glob(ext):
                try:
                    img = _read_image(img_path, grayscale=grayscale)
                    img = _postprocess_image(
                        img,
                        size=size,
                        enhance_contrast=enhance_contrast,
                        denoise=denoise,
                    )
                    if img is None:
                        print(f"Skip invalid image: {img_path}")
                        continue

                    out_path = out_class_dir / img_path.name
                    cv2.imwrite(str(out_path), img)
                    class_count += 1
                    total_images += 1
                except Exception as exc:
                    print(f"Error while processing {img_path}: {exc}")

        print(f"  -> processed {class_count} images")

    print(f"\nPreprocessing finished. Total images: {total_images}")
    print(f"Output dir: {processed_dir.resolve()}")
    return str(processed_dir)


def load_and_preprocess_dataset_from_annotations(
    images_root,
    annotations_dir,
    processed_dir,
    size=128,
    grayscale=True,
    roi_margin=0.08,
    enhance_contrast=True,
    denoise=True,
    min_box_size=6,
):
    """
    ROI-focused preprocessing for VOC-style annotations.
    Crops defect boxes from original images before resizing.
    """
    images_root = Path(images_root)
    annotations_dir = Path(annotations_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not images_root.exists():
        raise FileNotFoundError(f"Image directory not found: {images_root}")
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {annotations_dir}")

    xml_files = sorted(annotations_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML annotations found under: {annotations_dir}")

    total_crops = 0
    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as exc:
            print(f"Skip broken xml {xml_path.name}: {exc}")
            continue

        filename = root.findtext("filename")
        if not filename:
            print(f"Skip {xml_path.name}: missing filename")
            continue

        img_path = None
        direct = images_root / filename
        if direct.exists():
            img_path = direct
        else:
            for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
                candidate = images_root / Path(filename).stem / (Path(filename).stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path is None:
                matches = list(images_root.rglob(filename))
                if matches:
                    img_path = matches[0]

        if img_path is None or not img_path.exists():
            print(f"Skip {xml_path.name}: image not found ({filename})")
            continue

        img = _read_image(img_path, grayscale=grayscale)
        if img is None:
            print(f"Skip unreadable image: {img_path}")
            continue

        img_h, img_w = img.shape[:2]
        objects = root.findall("object")
        if not objects:
            continue

        for obj_idx, obj in enumerate(objects):
            class_name = obj.findtext("name", default="unknown").strip().replace(" ", "_")
            box = obj.find("bndbox")
            if box is None:
                continue

            try:
                xmin = int(float(box.findtext("xmin", "0")))
                ymin = int(float(box.findtext("ymin", "0")))
                xmax = int(float(box.findtext("xmax", "0")))
                ymax = int(float(box.findtext("ymax", "0")))
            except ValueError:
                continue

            xmin = max(0, min(xmin, img_w - 1))
            xmax = max(0, min(xmax, img_w - 1))
            ymin = max(0, min(ymin, img_h - 1))
            ymax = max(0, min(ymax, img_h - 1))
            box_w = xmax - xmin
            box_h = ymax - ymin
            if box_w < min_box_size or box_h < min_box_size:
                continue

            margin_x = int(box_w * roi_margin)
            margin_y = int(box_h * roi_margin)
            crop_x1 = max(0, xmin - margin_x)
            crop_y1 = max(0, ymin - margin_y)
            crop_x2 = min(img_w, xmax + margin_x)
            crop_y2 = min(img_h, ymax + margin_y)
            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                continue

            crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
            crop = _postprocess_image(
                crop,
                size=size,
                enhance_contrast=enhance_contrast,
                denoise=denoise,
            )
            if crop is None:
                continue

            out_class_dir = processed_dir / class_name
            out_class_dir.mkdir(parents=True, exist_ok=True)
            out_name = f"{Path(filename).stem}_obj{obj_idx:02d}.png"
            out_path = out_class_dir / out_name
            cv2.imwrite(str(out_path), crop)
            total_crops += 1

    if total_crops == 0:
        raise RuntimeError("No ROI crops were produced. Check annotation/image paths and labels.")

    print(f"\nROI preprocessing finished. Total crops: {total_crops}")
    print(f"Output dir: {processed_dir.resolve()}")
    return str(processed_dir)
