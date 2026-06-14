#!/usr/bin/env python
"""
Simple helper that extracts **verbatim** text from the PDF shipped in this repo.

``python scripts/extract_text.py`` will create a new file named
``trading_and_hedging_local_volatility.txt`` containing every page – prefixed with its
page number.  The script uses :mod:`pypdf`, which is already part of the virtual
environment.

The implementation is intentionally minimal and free of external dependencies
other than *pypdf*.
"""

from pathlib import Path

try:
    from pypdf import PdfReader
except Exception as exc:  # pragma: no cover – defensive guard
    raise RuntimeError("pypdf is required for this script") from exc

PDF_FILE = Path("Trading and Hedging Local Volatility.pdf")
OUTPUT_TEXT = Path("trading_and_hedging_local_volatility.txt")


def main() -> None:
    if not PDF_FILE.exists():  # pragma: no cover – defensive guard
        raise FileNotFoundError(f"{PDF_FILE!s} does not exist. Run from repo root.")

    reader = PdfReader(PDF_FILE)
    with OUTPUT_TEXT.open("w", encoding="utf-8") as out:
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            out.write(f"==== PAGE {i} ====\n{text}\n\n")

    print(f"Extracted PDF to: {OUTPUT_TEXT.resolve()} ({len(reader.pages)} pages)")


if __name__ == "__main__":  # pragma: no cover
    main()