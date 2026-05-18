"""Export a Jupyter notebook to PDF on Windows + Python 3.14.

Workaround for nbconvert's ``webpdf`` exporter, which uses Playwright's async
API from a worker thread and fails on Windows + Python 3.14 with
``NotImplementedError`` from ``loop.subprocess_exec`` (the worker thread gets a
``SelectorEventLoop``, which can't spawn subprocesses on Windows). This script
renders HTML with nbconvert and prints to PDF via Playwright's sync API.

Usage:
    python scripts/export_notebook_pdf.py path/to/notebook.ipynb [output.pdf]
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from nbconvert import HTMLExporter
from playwright.sync_api import sync_playwright


def export(notebook: Path, output: Path) -> None:
    html_exporter = HTMLExporter()
    html_exporter.template_name = "lab"
    body, _ = html_exporter.from_filename(str(notebook))

    with tempfile.TemporaryDirectory() as tmp:
        html_path = Path(tmp) / "notebook.html"
        html_path.write_text(body, encoding="utf-8")

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(html_path.as_uri(), wait_until="networkidle")
            page.pdf(
                path=str(output),
                format="A4",
                print_background=True,
                margin={"top": "15mm", "bottom": "15mm", "left": "12mm", "right": "12mm"},
            )
            browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()

    notebook = args.notebook.resolve()
    output = (args.output or notebook.with_suffix(".pdf")).resolve()

    export(notebook, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
