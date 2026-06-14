#!/usr/bin/env python
"""Convert the plain‑text dump of *Trading and Hedging Local Volatility* into a
well‑structured Markdown file.

The output is written to ``trading_and_hedging_local_volatility.md`` in the repo
root.  Each logical page becomes an H2 heading and the raw text from that page is
preserved verbatim.  Additionally, any sentence that contains both the word
*volatility* (case‑insensitive) **and** a variant of *gadget* (*gadget* or
*gadgets*) is extracted into a bullet list under a dedicated subheading.

Running this script does not require any external dependencies beyond
``pypdf`` (already installed in the virtual environment).  Once the Markdown
is generated you can paste it straight into documentation files, GitHub README,
or any static site generator.
"""

from __future__ import annotations

import re
from pathlib import Path

TEXT_FILE = Path("trading_and_hedging_local_volatility.txt")
OUTPUT_MARKDOWN = Path("trading_and_hedging_local_volatility.md")


def extract_pages(txt_path: Path) -> list[tuple[int, str]]:
    """Parse the TXT dump into a list of `(page_no, page_text)` tuples.

    The source file uses ``==== PAGE X ====`` as delimiters (produced by
    :mod:`pypdf`).  This function keeps everything between those markers.
    """

    with txt_path.open("r", encoding="utf-8") as f:
        content = f.read()
    pages: list[tuple[int, str]] = []
    parts = re.split(r"====\s*PAGE\s+(\d+)\s*=+=+", content)
    # The split yields an initial empty string followed by alternating
    # page_no / page_text pairs.  We consume them two at a time.
    it = iter(parts)
    next(it)  # skip leading text before first header (should be empty)
    for page_no_str, page_body in zip(it, it):
        try:
            page_no = int(page_no_str.strip())
        except ValueError:
            continue
        pages.append((page_no, page_body.rstrip()))
    return pages


def find_gadget_sentences(text: str) -> list[str]:
    """Return sentences that refer to volatility gadgets.

    The heuristic looks for sentences containing the word *volatility* and a
    variant of *gadget*.  Sentences are split on punctuation marks.
    """

    sentiment_pattern = re.compile(r"\. |\? |! |")
    sentences = [s.strip() for s in sentiment_pattern.split(text) if s]
    matches: list[str] = []
    for sent in sentences:
        if re.search(r"volatility", sent, re.I) and re.search(r"gadget[s]?", sent, re.I):
            matches.append(sent)
    return matches


def main() -> None:
    if not TEXT_FILE.exists():  # pragma: no cover – defensive guard
        raise FileNotFoundError(f"{TEXT_FILE!s} missing; run the extraction script first.")

    pages = extract_pages(TEXT_FILE)

    with OUTPUT_MARKDOWN.open("w", encoding="utf-8") as md:
        # Page‑0 is a placeholder for introductory title page if present.
        md.write("# Trading and Hedging Local Volatility\n\n")
        md.write(f"*Generated on {Path().absolute()}*\n\n---\n\n")

        for page_no, body in pages:
            md.write(f"## Page {page_no}\n\n")
            # Preserve raw text as a normal paragraph block.
            md.write(body + "\n\n")

            gadget_matches = find_gadget_sentences(body)
            if gadget_matches:
                md.write("### Volatility Gadget References\n\n")
                for sent in gadget_matches:
                    md.write(f"- {sent}\n")
                md.write("\n---\n\n")

    print(f"Markdown report written to: {OUTPUT_MARKDOWN.resolve()}")


if __name__ == "__main__":  # pragma: no cover
    main()