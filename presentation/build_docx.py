"""Convert presentation/report.md → presentation/report.docx via pandoc.

Run:  .venv/bin/python presentation/build_docx.py

Pandoc handles tables, images, code blocks, and YAML front-matter natively,
so the docx looks structurally similar to the PDF and is fully editable in
Word / Google Docs / LibreOffice.
"""

from pathlib import Path

import pypandoc

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "report.md"
DST = ROOT / "report.docx"

# Resource path tells pandoc where to resolve relative image paths
# (our images live in presentation/plots/...).
pypandoc.convert_file(
    str(SRC),
    "docx",
    format="markdown+yaml_metadata_block+pipe_tables+fenced_code_blocks",
    outputfile=str(DST),
    extra_args=[
        f"--resource-path={ROOT}",
        "--toc",
        "--toc-depth=2",
        "--standalone",
    ],
)

print(f"wrote {DST}  ({DST.stat().st_size / 1024:.1f} KB)")
