# Latte presentation

Latte borrows the visual language from KeVin's October 22 implementation:
Architects Daughter, quiet blue-gray controls, role-colored matrices, thin
contours, graph-paper formula cards, and blue operand ribbons. Latte's own
training, forward pass, recorded losses, gradients, and AdamW updates supply
all of the data. Its model architecture and datasets are unchanged.

## Reading the display

- Weights are gray before an update and warm orange after applying it.
- Keys are green; values purple; queries muted blue (`#8FA7D8`); attention cyan; other
  activations yellow; recorded losses warm red.
- Shading is normalized within each matrix. It is not a shared numeric scale.
  Signs remain available in the numerical labels, full-precision hover values,
  and weight inspector.
- Numbers starts enabled. Matrix values appear at detail zoom; the overview
  continues to show the matrix topology. Turning Numbers off hides cell labels
  without altering values or calculations.
- Math starts disabled. Enable it, zoom into a matrix, and select a cell to
  inspect its formula. Blue ribbons identify the actual operand row/column.
  Scalars use letters and vectors use single bars, for example `a = |b| · |c|`.
  Ribbons use KeVin's lifted tape geometry, matrix-edge anchors, full-width
  landings, and translucent gradients. Each operand row or column has one
  continuous contour; individual cells have no selection outlines.
  Ribbons and contours share the innermost visible bracket's content clip,
  with KeVin's short feathered margin. Floating operand badges use the same
  content boundary for placement: a row badge moves to the right end and points
  left when the left margin is clipped, then follows the ribbon if needed.
  Operand badges use KeVin's placement rules: visible row margins point right
  into the row; when panning hides that margin, the badge follows the operand
  edge or ribbon and turns toward the offscreen input.
- Matrix variables (Q, K, V, Wq, S, A, and the other training tensors) sit at
  their matrices' centers in overview mode. When numbers are visible, name
  plates dock immediately above their own column indices, centered horizontally.
  At lower numbered zooms, the text, padding, and corners scale together to
  keep the plate clear of the preceding matrix.
  Matrix names are clipped at the same content boundary as the matrix ink;
  hierarchy captions remain in the margin.
  Clicking a name provides its matrix-level equation with Math enabled; without Math it focuses
  the matrix. Name selections connect the related matrix names with curved
  lines and keep those matrices visible; they do not use scalar letter badges
  or cell ribbons. Cell selections retain their separate operand-ribbon view.
  Connections to offscreen matrices continue toward their actual matrix
  label positions through the clipped viewport edge, even when their names are culled.
  During a selection, unrelated matrices retain 40% opacity; matrix names retain
  60% during cell selection and 85% during matrix-name selection,
  and operation captions 50%; connected matrix names remain fully visible.
  Names use KeVin's compact pale label plate: 2px vertical and 4px horizontal
  padding, 3px corners, a translucent `#edf1f6e8` background, and no text halo.
  The plate shares the name's selection dimming and bracket clipping; its
  padding is clickable too. Hovering or focusing a name in Math mode previews its matrix
  with a quiet blue outline and gently restores its colors without selecting
  it or changing the current formula. Leaving the name removes the preview.
  Escape, Clear selection, or selecting the same cell dismisses the card.
  Escape also releases the stage focus left by a cell click, preventing a
  whole-stage focus outline after deselection. Keyboard focus indicators remain.
  Disabling Math clears it. Wheel gestures over a card pan the diagram; Ctrl/Command
  plus wheel zooms. Operand badges move the camera to that input.
- The weight inspector keeps its recorded numbers and adds symbolic AdamW
  equations when Math is enabled. Optimizer moments are symbolic; no moment
  values are fabricated or inferred from rounded displayed weights.

The training step, sequence, parameter, row/column selectors, before/after
toggle, corpus location, resizable sidebar sections, and camera controls retain
their previous functions. Existing one-based row/column controls and matrix
indices are retained. The prediction table and recorded losses always describe
the recorded state before the update, including while the diagram shows the
recomputed state after the update.

## Appearance configuration

Edit `appearance.css` to change font families, label sizes, formula sizes,
and borders. The bundled font and its SIL Open Font License live in `fonts/`.
The exporter embeds the font so standalone HTML also works offline.

Regenerate an existing export after editing appearance or renderer files:

```bash
.venv/bin/python export_trace_html.py \
  --trace out/default/training.trace.json \
  --output out/default/latte.html --show training
```

This reuses the recording and weights; it does not retrain the model. Refresh
the local app to see the regenerated shell. React source changes additionally
require `npm run build` in `frontend`.

`presentation_math.js` describes formulas from Latte's actual parsed expression
trees. `presentation.js` draws formula cards, outlines, labels, and operand
links. The existing formula evaluator still computes the forward pass.
The SVG outlines and ribbons use keyed D3 joins, retaining paths, gradients,
and filters during camera movement. Annotation labels keep their identity by
matrix name, and their completed camera update synchronously positions the
connectors before the browser paints. The bundled `vendor/d3-selection.min.js`
also makes this work in standalone exports without a network connection;
its license is included alongside it.
Diagram-picking controls are positioned only while picking is active. Active
picking indexes the visible operation labels once per update, avoiding a full
document search for every stage in larger architectures such as `99999`.
Training replay evaluates the full forward pass on step and sequence changes,
but queues offscreen cell text and colors until their stage becomes visible.
The camera flushes those updates before paint. Unchanged text and matrix color
ranges are reused, and value updates retain the existing matrix-name index.
Grayscale filters and cell transitions are also limited to visible stages, so
applying an update does not allocate effects for every offscreen weight cell.

The previous presentation sources and compiled React interface are preserved in
`backups/before-kevin-presentation/`. That snapshot is excluded from packaged
releases. It does not contain or replace any training data.
