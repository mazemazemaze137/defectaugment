from __future__ import annotations

from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "毕业论文_初稿_格式化.pdf"
OUT_DIR = ROOT / "docs" / "rendered_thesis_docx"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("page-*.png"):
        old.unlink()
    pdf = fitz.open(PDF_PATH)
    for index, page in enumerate(pdf, start=1):
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        pix.save(OUT_DIR / f"page-{index:02d}.png")
    print(f"rendered {len(pdf)} pages to {OUT_DIR}")


if __name__ == "__main__":
    main()
