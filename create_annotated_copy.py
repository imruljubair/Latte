"""Create a separately named explorer with margin annotations."""
from pathlib import Path
import argparse
import base64


def create_annotated_copy(source: Path, destination: Path | None = None) -> Path:
    root = Path(__file__).resolve().parent
    source = Path(source).resolve()
    destination = Path(destination).resolve() if destination else source.with_name(f'{source.stem}-annotated.html')
    if destination == source:
        raise ValueError('Annotated output must differ from the source HTML')
    html = source.read_text(encoding='utf-8')
    worked_script = (root / 'worked_calculations.js').read_text(encoding='utf-8')
    marker = '  function renderCalculationDetails(cell) {'
    if marker not in html:
        raise ValueError('Could not locate calculation renderer')
    html = html.replace(marker, worked_script + '\n' + marker + '\n    return renderWorkedCalculation(cell);', 1)
    # Increase curvature in both directions in this copy only.
    original_path = 'path.setAttribute("d", `M ${source.x} ${source.y} C ${source.x + bend * direction} ${source.y}, ${end.x - bend * direction} ${end.y}, ${end.x} ${end.y}`);'
    curved_path = '''const dx = end.x - source.x;
      const dy = end.y - source.y;
      const distance = Math.hypot(dx, dy);
      const arc = Math.min(70, Math.max(18, distance * .18));
      const normalX = distance ? dy / distance * arc : 0;
      const normalY = distance ? -dx / distance * arc : 0;
      const curveBend = Math.max(45, Math.abs(dx) * .45);
      path.setAttribute("d", `M ${source.x} ${source.y} C ${source.x + curveBend * direction + normalX} ${source.y + normalY}, ${end.x - curveBend * direction + normalX} ${end.y + normalY}, ${end.x} ${end.y}`);'''
    if original_path not in html:
        raise ValueError('Could not locate the dependency connection path')
    html = html.replace(original_path, curved_path, 1)
    css = (root / 'annotation_overlay.css').read_text(encoding='utf-8')
    script = (root / 'annotation_overlay.js').read_text(encoding='utf-8')
    zoom_script = (root / 'annotated_zoom.js').read_text(encoding='utf-8')
    font = base64.b64encode((root / 'fonts/ArchitectsDaughter-Regular.ttf').read_bytes()).decode('ascii')
    appearance = (root / 'appearance.css').read_text(encoding='utf-8').replace('__LATTE_FONT__', 'data:font/ttf;base64,' + font)
    brand_font = base64.b64encode((root / 'fonts/Fraunces-Variable.ttf').read_bytes()).decode('ascii')
    appearance = appearance.replace('__LATTE_BRAND_FONT__', 'data:font/ttf;base64,' + brand_font)
    logo = 'data:image/png;base64,' + base64.b64encode((root / 'assets/latte-logo.png').read_bytes()).decode('ascii')
    appearance = appearance.replace('__LATTE_LOGO__', logo)
    html = html.replace('</head>', f'<link rel="icon" type="image/png" href="{logo}">\n</head>', 1)
    math_script = (root / 'presentation_math.js').read_text(encoding='utf-8')
    ribbon_script = (root / 'ribbon_geometry.js').read_text(encoding='utf-8')
    d3_script = (root / 'vendor/d3-selection.min.js').read_text(encoding='utf-8')
    presentation_script = (root / 'presentation.js').read_text(encoding='utf-8')
    usage_script = (root / 'frontend/src/corpus-usage.mjs').read_text(encoding='utf-8').replace('export function', 'function')
    usage_script = '(function(){' + usage_script + '\nwindow.LatteCorpusUsage={usageAt,usageColor,usageSegments};})();'
    html = html.replace('</head>', f'<style>\n{appearance}\n</style>\n<script>\n{d3_script}\n</script>\n<script>\n{math_script}\n</script>\n<script>\n{ribbon_script}\n</script>\n</head>', 1)
    html = html.replace('</head>', f'<style>\n{css}\n</style>\n</head>', 1)
    html = html.replace('</body>', f'<script>\n{script}\n</script>\n<script>\n{zoom_script}\n</script>\n<script>\n{usage_script}\n</script>\n<script>\n{presentation_script}\n</script>\n</body>', 1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding='utf-8')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(create_annotated_copy(args.source, args.output))


if __name__ == '__main__':
    main()
