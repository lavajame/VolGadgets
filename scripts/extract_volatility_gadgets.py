#!/usr/bin/env python
"""
Extract sections from the "Trading and Hedging Local Volatility" PDF that reference the
concept of *volatility gadgets*.

The script uses :class:`pypdf.PdfReader` to read the local PDF file.  It scans every
page for lines that contain both the word ``volatility`` (case‑insensitive) and an
instance of the noun *gadget* (again case‑insensitive).  Matching sentences are
collected and written out as a lightweight Markdown document named
``volatility_gadgets_summary.md`` in the repository root.

To run this script you need to activate the project virtual environment first:

>>> .\venv\Scripts\activate   # Windows 10/11 PowerShell or cmd, Unix ``source venv/bin/activate``
>>> python scripts/extract_volatility_gadgets.py

The generated Markdown can then be reviewed or added to the documentation.
"""

from __future__ import annotations

import re
from pathlib import Path

try:
    # The project uses ``pypdf`` – we assume it's available in the venv.
    from pypdf import PdfReader
except ImportError as exc:  # pragma: no cover - defensive, should never happen here
    raise RuntimeError(
        "The 'pypdf' package is required. Install via pip or ensure your virtual environment is activated."
    ) from exc


PDF_FILE = Path("Trading and Hedging Local Volatility.pdf")
OUTPUT_MD = Path("volatility_gadgets_summary.md")


def extract_volatility_gadget_sentences(reader: PdfReader) -> list[tuple[int, str]]:
    """Return a list of (page_no, sentence) tuples.

    Each entry contains the 1‑based page number and the sentence that mentions
    both the word *volatility* and *gadget*.  The search is performed on a line
    basis – this keeps the output concise while still preserving context.
    """

    sentences: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        for line in text.splitlines():
            if re.search(r"\bvolatility\b", line, re.I) and re.search(r"gadget", line, re.I):
                sentences.append((i + 1, line.strip()))
    return sentences


def main() -> None:
    if not PDF_FILE.exists():  # pragma: no cover - safety check
        raise FileNotFoundError(f"{PDF_FILE} does not exist – run from the repository root.")

    reader = PdfReader(PDF_FILE)
    matches = extract_volatility_gadget_sentences(reader)

    if not matches:
        content = "# Volatility Gadgets\n\nNo mentions of volatility gadgets were found in the PDF."
    else:
        content_lines: list[str] = ["# Volatility Gadgets", ""]
        for page_no, sentence in matches:
            content_lines.append(f"## Page {page_no}")
            content_lines.append(f"- {sentence}")
            content_lines.append("")
        content = "\n".join(content_lines)

    OUTPUT_MD.write_text(content, encoding="utf8")
    print(f"Markdown written to: {OUTPUT_MD.resolve()}")


if __name__ == "__main__":  # pragma: no cover - standard guard
    main()