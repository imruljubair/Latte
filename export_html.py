from __future__ import annotations

import argparse
from html import escape, unescape
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile
from xml.etree import ElementTree


SHEET_PATTERN = re.compile(
    r'<A NAME="table(?P<index>\d+)"><h1>Sheet \d+: '
    r'<em>(?P<name>.*?)</em></h1></A>\s*'
    r'(?P<table><table\b.*?</table>)',
    re.IGNORECASE | re.DOTALL,
)
COLGROUP_PATTERN = re.compile(
    r"<colgroup(?:\s+span=\"?(?P<span>\d+)\"?)?\s+width=\"(?P<width>\d+)\"[^>]*>",
    re.IGNORECASE,
)
RUNTIME_PATH = Path(__file__).with_name("html_runtime.js")


def find_soffice(explicit: str | None = None) -> str:
    candidates = [
        explicit,
        shutil.which("soffice"),
        shutil.which("libreoffice"),
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise FileNotFoundError(
        "LibreOffice is required for HTML export. Install LibreOffice or pass "
        "--soffice PATH."
    )


def extract_style(document: str) -> str:
    match = re.search(
        r"<style\b[^>]*>(?P<style>.*?)</style>",
        document,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group("style").strip() if match else ""


def lock_table_to_exported_column_widths(table: str) -> str:
    width = sum(
        int(match.group("span") or "1") * int(match.group("width"))
        for match in COLGROUP_PATTERN.finditer(table)
    )
    if width == 0:
        return table
    return re.sub(
        r"<table\b",
        f'<table style="table-layout:fixed;width:{width}px"',
        table,
        count=1,
        flags=re.IGNORECASE,
    )


def excel_column_number(label: str) -> int:
    number = 0
    for character in label.upper():
        number = number * 26 + ord(character) - ord("A") + 1
    return number


def workbook_grid_offsets(
    workbook_path: Path,
    _document: str,
) -> tuple[dict[str, int], dict[str, int]]:
    main_namespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    relationship_namespace = (
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    )
    package_namespace = "http://schemas.openxmlformats.org/package/2006/relationships"
    column_offsets: dict[str, int] = {}
    row_offsets: dict[str, int] = {}
    with zipfile.ZipFile(workbook_path) as archive:
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        relationships = ElementTree.fromstring(
            archive.read("xl/_rels/workbook.xml.rels")
        )
        targets = {
            relationship.attrib["Id"]: relationship.attrib["Target"]
            for relationship in relationships.findall(
                f"{{{package_namespace}}}Relationship"
            )
        }
        for sheet in workbook.findall(f".//{{{main_namespace}}}sheet"):
            relationship_id = sheet.attrib[f"{{{relationship_namespace}}}id"]
            target = targets[relationship_id].lstrip("/")
            if not target.startswith("xl/"):
                target = f"xl/{target}"
            worksheet = ElementTree.fromstring(archive.read(target))
            addresses = [
                cell.attrib.get("r", "")
                for cell in worksheet.findall(f".//{{{main_namespace}}}c")
                if cell.find(f"{{{main_namespace}}}v") is not None
                or cell.find(f"{{{main_namespace}}}f") is not None
                or cell.find(f"{{{main_namespace}}}is") is not None
            ]
            first_column = min(
                (
                    excel_column_number(match.group(1))
                    for address in addresses
                    if (match := re.match(r"([A-Z]+)", address))
                ),
                default=1,
            )
            first_row = min(
                (
                    int(match.group(1))
                    for address in addresses
                    if (match := re.search(r"(\d+)$", address))
                ),
                default=1,
            )
            column_offsets[sheet.attrib["name"]] = first_column - 1
            row_offsets[sheet.attrib["name"]] = first_row - 1
    return column_offsets, row_offsets


def build_workbook_html(
    document: str,
    title: str,
    *,
    column_offsets: dict[str, int] | None = None,
    row_offsets: dict[str, int] | None = None,
) -> str:
    sheets = [
        {
            "index": int(match.group("index")),
            "name": unescape(re.sub(r"<[^>]+>", "", match.group("name"))),
            "table": lock_table_to_exported_column_widths(match.group("table")),
        }
        for match in SHEET_PATTERN.finditer(document)
    ]
    if not sheets:
        raise ValueError("LibreOffice HTML did not contain any worksheet tables")

    source_style = extract_style(document)
    explorer_style = Path(__file__).with_name("explorer.css").read_text(encoding="utf-8")
    runtime_script = RUNTIME_PATH.read_text(encoding="utf-8")
    sheet_sections = []
    tab_buttons = []
    for position, sheet in enumerate(sheets):
        active_class = " active" if position == 0 else ""
        selected = "true" if position == 0 else "false"
        sheet_id = f"sheet-{sheet['index']}"
        column_offset = (column_offsets or {}).get(sheet["name"], 0)
        row_offset = (row_offsets or {}).get(sheet["name"], 0)
        sheet_sections.append(
            f'<section class="worksheet{active_class}" id="{sheet_id}" '
            f'aria-label="{escape(sheet["name"], quote=True)}" '
            f'data-column-offset="{column_offset}" '
            f'data-row-offset="{row_offset}">'
            f'{sheet["table"]}</section>'
        )
        tab_buttons.append(
            f'<button class="sheet-tab{active_class}" type="button" '
            f'data-sheet="{sheet_id}" aria-selected="{selected}">'
            f'{escape(sheet["name"])}</button>'
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
{source_style}
    :root {{ color-scheme: light; }}
    html, body {{ margin: 0; min-height: 100%; background: #fff; }}
    body {{ overflow: auto; padding-bottom: 31px; }}
    font[face="Ink Free"] {{
      font-family: "Ink Free", "Comic Sans MS", "Comic Sans", cursive !important;
    }}
    font[face="Arial"] {{ font-family: Arial, Helvetica, sans-serif !important; }}
    .worksheet td > font[size="4"] {{
      position: relative;
      z-index: 2;
      display: inline-block;
      white-space: nowrap;
    }}
    .worksheet {{
      display: none;
      width: max-content;
      min-width: 100%;
      zoom: var(--workbook-zoom, 1);
    }}
    .worksheet.active {{ display: block; }}
    .worksheet > table {{ margin: 0; table-layout: fixed; }}
    .sheet-tabs {{
      position: fixed;
      z-index: 20;
      left: 0;
      right: 0;
      bottom: 0;
      height: 30px;
      display: flex;
      align-items: end;
      gap: 1px;
      padding: 0 8px;
      overflow-x: auto;
      background: #f3f3f3;
      border-top: 1px solid #d0d0d0;
      box-sizing: border-box;
    }}
    .sheet-tab {{
      height: 29px;
      padding: 0 14px;
      border: 0;
      border-bottom: 3px solid transparent;
      background: transparent;
      color: #222;
      font: 12px Arial, sans-serif;
      white-space: nowrap;
      cursor: pointer;
    }}
    .sheet-tab:hover {{ background: #e8e8e8; }}
    .sheet-tab.active {{
      background: #fff;
      border-bottom-color: #d00000;
      font-weight: 600;
    }}
    .live-input-token {{ cursor: text; }}
    .live-input-token:focus {{
      outline: 2px solid #217346;
      outline-offset: -2px;
    }}
    .workbook-toolbar, .sequence-editor {{
      position: fixed;
      z-index: 60;
      top: 12px;
      display: flex;
      align-items: center;
      gap: 5px;
      min-height: 36px;
      padding: 5px 7px;
      border: 1px solid rgba(0, 0, 0, .14);
      border-radius: 10px;
      background: rgba(255, 255, 255, .94);
      box-shadow: 0 5px 18px rgba(0, 0, 0, .14);
      backdrop-filter: blur(10px);
      box-sizing: border-box;
      font: 12px Arial, sans-serif;
    }}
    .workbook-toolbar {{ right: 12px; }}
    .sequence-editor {{ left: 12px; }}
    .sequence-editor.hidden {{ display: none; }}
    .toolbar-group {{ display: flex; align-items: center; gap: 3px; }}
    .toolbar-divider {{ width: 1px; height: 22px; background: #ddd; margin: 0 2px; }}
    .toolbar-button, .math-close, .block-toggle {{
      border: 0;
      border-radius: 6px;
      background: #f2f3f5;
      color: #202124;
      font: 600 12px Arial, sans-serif;
      cursor: pointer;
    }}
    .toolbar-button {{ min-width: 29px; height: 28px; padding: 0 8px; }}
    .toolbar-button:hover, .math-close:hover, .block-toggle:hover {{ background: #e2e6e9; }}
    .zoom-label {{ min-width: 76px; text-align: center; color: #3c4043; }}
    .sequence-title {{ margin: 0 4px 0 2px; font-weight: 700; color: #3c4043; }}
    .sequence-token {{
      width: 58px;
      height: 27px;
      padding: 0 5px;
      border: 1px solid #c8cdd2;
      border-radius: 6px;
      background: #fff;
      color: #111;
      text-align: center;
      font: 15px "Ink Free", "Comic Sans MS", cursive;
      box-sizing: border-box;
    }}
    .sequence-token:focus {{ border-color: #217346; outline: 2px solid rgba(33,115,70,.18); }}
    .block-toggle {{
      width: 22px;
      height: 22px;
      margin: 0 5px 0 2px;
      padding: 0;
      vertical-align: middle;
      color: #555;
    }}
    tr.block-header {{ cursor: pointer; transition: filter .12s ease; }}
    tr.block-header:hover {{ filter: brightness(.97); }}
    tr.block-header.block-related {{ filter: drop-shadow(0 0 4px rgba(47,128,237,.8)); }}
    td.calculation-cell {{ cursor: help; transition: box-shadow .12s ease; }}
    td.calculation-cell:hover {{ box-shadow: inset 0 0 0 2px rgba(33,115,70,.55); }}
    td.math-selected {{ box-shadow: inset 0 0 0 3px #217346 !important; }}
    td.math-dependency {{ box-shadow: inset 0 0 0 2px #2f80ed !important; }}
    .dependency-overlay {{
      position: absolute;
      z-index: 45;
      top: 0;
      left: 0;
      pointer-events: none;
      overflow: visible;
    }}
    .dependency-link {{
      fill: none;
      stroke: #2f80ed;
      stroke-width: 2;
      stroke-linecap: round;
      opacity: .78;
      vector-effect: non-scaling-stroke;
    }}
    .dependency-link-dot {{ fill: #2f80ed; opacity: .9; }}
    .math-panel {{
      position: fixed;
      z-index: 80;
      top: 0;
      right: 0;
      bottom: 31px;
      width: min(390px, 92vw);
      padding: 18px;
      overflow: auto;
      border-left: 1px solid #d9dde1;
      background: rgba(255,255,255,.98);
      box-shadow: -8px 0 28px rgba(0,0,0,.16);
      transform: translateX(105%);
      transition: transform .22s ease;
      box-sizing: border-box;
      font: 13px/1.45 Arial, sans-serif;
    }}
    .math-panel.open {{ transform: translateX(0); }}
    .math-panel-header {{ display: flex; align-items: start; justify-content: space-between; gap: 12px; }}
    .math-panel h2 {{ margin: 0; font-size: 18px; color: #1f252b; }}
    .math-panel h3 {{ margin: 18px 0 7px; font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: #687078; }}
    .math-close {{ width: 30px; height: 30px; font-size: 17px; }}
    .math-result {{ margin-top: 10px; padding: 12px; border-radius: 9px; background: #edf7f1; color: #145c35; font-size: 20px; font-weight: 700; }}
    .math-formula {{ display: block; padding: 10px; overflow-wrap: anywhere; border-radius: 8px; background: #f5f6f7; color: #262b30; font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .math-functions {{ display: flex; flex-wrap: wrap; gap: 5px; margin-top: 9px; }}
    .math-chip {{ padding: 3px 7px; border-radius: 999px; background: #e8f0fe; color: #174ea6; font-weight: 700; }}
    .math-inputs {{ display: grid; gap: 5px; }}
    .math-input {{ display: grid; grid-template-columns: minmax(110px, 1fr) 1fr; gap: 8px; padding: 6px 8px; border-radius: 6px; background: #fafafa; }}
    .math-input code {{ overflow-wrap: anywhere; color: #245da8; }}
    .math-input span {{ text-align: right; overflow-wrap: anywhere; }}
    body.color-overview .worksheet.active td,
    body.color-overview .worksheet.active font,
    body.color-overview .worksheet.active a {{ color: transparent !important; text-shadow: none !important; }}
    body.color-overview .worksheet.active .block-toggle {{ visibility: hidden; }}
    @media print {{
      body {{ padding-bottom: 0; }}
      .sheet-tabs, .workbook-toolbar, .sequence-editor, .math-panel, .block-toggle, .dependency-overlay {{ display: none !important; }}
      .worksheet {{ display: block; break-after: page; }}
    }}
{explorer_style}
  </style>
</head>
<body>
  {''.join(sheet_sections)}
  <nav class="sheet-tabs" aria-label="Worksheets">
    {''.join(tab_buttons)}
  </nav>
  <script>
    const worksheets = [...document.querySelectorAll('.worksheet')];
    const tabs = [...document.querySelectorAll('.sheet-tab')];
    function showSheet(id, updateHash = true) {{
      if (!worksheets.some((sheet) => sheet.id === id)) return;
      worksheets.forEach((sheet) => sheet.classList.toggle('active', sheet.id === id));
      tabs.forEach((tab) => {{
        const active = tab.dataset.sheet === id;
        tab.classList.toggle('active', active);
        tab.setAttribute('aria-selected', String(active));
      }});
      if (updateHash && window.location.href !== 'about:srcdoc') {{
        const url = new URL(location.href);
        url.hash = id === 'sheet-0' ? '' : id;
        history.replaceState(null, '', url);
      }}
      window.scrollTo(0, 0);
    }}
    tabs.forEach((tab) => tab.addEventListener('click', () => showSheet(tab.dataset.sheet)));
    if (location.hash) showSheet(location.hash.slice(1), false);
    if (location.hash === '#sheet-0') {{
      const url = new URL(location.href); url.hash = '';
      history.replaceState(null, '', url);
    }}
  </script>
  <script>
{runtime_script}
  </script>
</body>
</html>
"""


def export_workbook_html(
    workbook_path: Path,
    output_path: Path,
    *,
    soffice: str | None = None,
) -> Path:
    workbook_path = workbook_path.resolve()
    output_path = output_path.resolve()
    if not workbook_path.is_file():
        raise FileNotFoundError(f"Workbook not found: {workbook_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    executable = find_soffice(soffice)
    with tempfile.TemporaryDirectory(prefix="latte-html-") as temporary:
        temporary_path = Path(temporary)
        result = subprocess.run(
            [
                executable,
                "--headless",
                "--convert-to",
                "html",
                "--outdir",
                str(temporary_path),
                str(workbook_path),
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        converted_path = temporary_path / f"{workbook_path.stem}.html"
        if result.returncode != 0 or not converted_path.is_file():
            details = (result.stderr or result.stdout).strip()
            raise RuntimeError(
                "LibreOffice could not convert the workbook to HTML"
                + (f": {details}" if details else "")
            )
        converted = converted_path.read_text(encoding="utf-8")

    column_offsets, row_offsets = workbook_grid_offsets(workbook_path, converted)
    output_path.write_text(
        build_workbook_html(
            converted,
            workbook_path.stem,
            column_offsets=column_offsets,
            row_offsets=row_offsets,
        ),
        encoding="utf-8",
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an Excel workbook as an interactive, worksheet-faithful HTML file"
    )
    parser.add_argument("--workbook", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--soffice")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = export_workbook_html(
        Path(args.workbook),
        Path(args.output),
        soffice=args.soffice,
    )
    print(f"saved HTML: {output}")


if __name__ == "__main__":
    main()
