"""Convert presentation/report.md → presentation/report.pdf

Run:  .venv/bin/python presentation/build_pdf.py
"""

import re
from pathlib import Path

import markdown
from weasyprint import HTML, CSS

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "report.md"
DST = ROOT / "report.pdf"

text = SRC.read_text()

# Pull title/subtitle/author/date out of the YAML front-matter for the title block.
title = subtitle = author = date = ""
m = re.match(r"^---\n(.*?)\n---\n", text, flags=re.DOTALL)
if m:
    fm = m.group(1)
    for key in ("title", "subtitle", "author", "date"):
        km = re.search(rf'^{key}:\s*"?([^"\n]+)"?\s*$', fm, flags=re.MULTILINE)
        if km:
            globals().__setitem__(key, km.group(1).strip())
    text = text[m.end():]

title_block = f"""
<div class="title-block">
  <div class="title">{title}</div>
  <div class="subtitle">{subtitle}</div>
  <div class="byline">{author} · {date}</div>
</div>
""" if title else ""

html_body = title_block + markdown.markdown(
    text,
    extensions=["extra", "sane_lists", "toc", "tables", "fenced_code"],
)

CSS_RULES = """
@page { size: Letter; margin: 0.9in 0.9in 0.9in 0.9in; }
body { font-family: 'DejaVu Sans', 'Liberation Sans', sans-serif; font-size: 10.5pt; line-height: 1.45; color: #222; }
.title-block { text-align: center; margin: 8pt 0 30pt 0; padding-bottom: 12pt; border-bottom: 2px solid #444; }
.title-block .title { font-size: 22pt; font-weight: bold; margin-bottom: 6pt; }
.title-block .subtitle { font-size: 13pt; font-style: italic; color: #555; margin-bottom: 14pt; }
.title-block .byline { font-size: 11pt; color: #444; }
h1 { font-size: 18pt; border-bottom: 1.5px solid #444; padding-bottom: 4px; margin-top: 22px; }
h2 { font-size: 13pt; margin-top: 18px; color: #1a1a1a; }
h3 { font-size: 11.5pt; margin-top: 14px; color: #333; }
p, li { font-size: 10.5pt; }
table { border-collapse: collapse; margin: 10px 0; width: 100%; font-size: 9.5pt; }
th, td { border: 1px solid #999; padding: 4px 7px; text-align: left; }
th { background: #efefef; }
code { font-family: 'DejaVu Sans Mono', 'Liberation Mono', monospace; background: #f3f3f3;
       padding: 1px 3px; border-radius: 3px; font-size: 9pt; }
pre { background: #f3f3f3; padding: 8px; border-radius: 4px; font-size: 8.5pt;
      overflow-wrap: anywhere; white-space: pre-wrap; }
blockquote { border-left: 3px solid #888; margin-left: 0; padding-left: 12px; color: #444; }
img { max-width: 100%; height: auto; display: block; margin: 8px auto; }
hr { border: none; border-top: 1px solid #bbb; margin: 16px 0; }
"""

html_full = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>CS671 Team MVP report</title></head>
<body>{html_body}</body></html>"""

HTML(string=html_full, base_url=str(ROOT)).write_pdf(
    str(DST),
    stylesheets=[CSS(string=CSS_RULES)],
)
print(f"wrote {DST}")
