import argparse
import json
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _normalize_class_name(name):
    return name.strip().lower().replace("_", "-")


def _collect_classes(source_dir):
    source_dir = Path(source_dir)
    classes = {}
    for class_dir in sorted([p for p in source_dir.iterdir() if p.is_dir()]):
        paths = [
            path
            for path in sorted(class_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        classes[_normalize_class_name(class_dir.name)] = (class_dir.name, paths)
    return classes


def build_hybrid_dataset(sources, output_dir, max_per_class_per_source=50):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_maps = [(Path(source), _collect_classes(source)) for source in sources]
    class_keys = sorted(set().union(*(set(class_map.keys()) for _, class_map in source_maps)))
    summary = {
        "sources": [str(Path(source)) for source in sources],
        "output_dir": str(output_dir),
        "max_per_class_per_source": max_per_class_per_source,
        "classes": {},
    }

    for class_key in class_keys:
        out_class_name = None
        copied = 0
        details = {}
        for source_idx, (source_dir, class_map) in enumerate(source_maps):
            if class_key not in class_map:
                continue
            class_name, paths = class_map[class_key]
            out_class_name = out_class_name or class_name
            selected = paths[:max_per_class_per_source]
            out_class_dir = output_dir / out_class_name
            out_class_dir.mkdir(parents=True, exist_ok=True)
            for item_idx, src_path in enumerate(selected):
                out_name = f"s{source_idx}_{item_idx:04d}_{src_path.name}"
                shutil.copy2(src_path, out_class_dir / out_name)
            copied += len(selected)
            details[str(source_dir)] = {"available": len(paths), "copied": len(selected)}
        summary["classes"][out_class_name or class_key] = {"copied": copied, "sources": details}

    summary["total_copied"] = sum(item["copied"] for item in summary["classes"].values())
    (output_dir / "hybrid_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Build a class-balanced hybrid augmented dataset.")
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-per-class-per-source", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = build_hybrid_dataset(
        sources=args.sources,
        output_dir=args.output_dir,
        max_per_class_per_source=args.max_per_class_per_source,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
