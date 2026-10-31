# Latte

**Language-model Architecture, Training & Tuning Explorer**

**[Try the web demo](https://imruljubair.github.io/Latte/)** ·
**[Run locally](https://imruljubair.github.io/Latte/run-locally.html)**

The web demo uses the same Explorer and Training interface as the local app,
with a real recorded run. Editing the corpus or settings, starting training,
or deleting a run shows a Run locally message. Run locally to train with your own
corpus and architecture. See [web demo development and publication](docs/web-demo.md).

Latte is a local training explorer for a tiny educational transformer. Inspect
input sequences, next-token predictions, losses, gradients, and AdamW updates.
The React + TypeScript application uses D3 for prediction-table updates,
SVG overlays, and numerical helpers, with a FastAPI server backed by PyTorch. Its presentation
follows KeVin's handwriting typography, compact controls, role-colored matrices,
and graph-paper math cards. Latte keeps its own training model and workflow.
See [appearance and formula controls](docs/appearance.md).

## Run locally

Download this project, then run:

```bash
cd Latte
./start
```

On Windows, or without the shell launcher:

```bash
python launch.py
```

You need Python 3.11 or newer; Python 3.12 is the tested version. The launcher
creates Latte's own `.venv`, installs Python dependencies, records a default
10-step experiment if one does not exist, and opens `http://127.0.0.1:8000`.
The first setup requires internet access to install dependencies. Subsequent
training and visualization run locally without API keys or external services.
The server binds only to the local loopback interface. Stop it with Ctrl+C.

The prebuilt interface is included in `frontend/dist`, so users do not need
Node.js. Node.js 22 or newer is recommended when modifying or rebuilding the
frontend. If the prebuilt files are missing, the launcher builds them using npm.
You can select a Python interpreter with `LATTE_PYTHON=python3.12 ./start`.

```bash
./start --port 8080 --no-open
```

## Use the explorer

Select a recorded training step and batch sequence, inspect a weight, and click
**Apply update** to compare the forward pass after the recorded update. The
prediction table and displayed recorded batch loss describe the state before
that update, matching the standalone explorer.

Enable **Math** to inspect cell formulas and operand ribbons, or select a matrix
name for its equation. Math starts off; **Numbers** starts on. The weight
inspector retains its recorded numbers and adds symbolic AdamW equations while
Math is enabled. All formulas describe Latte's own training computations.

Use **Choose blocks** at the top of the canvas to select blocks from a list or
pick them directly on the diagram, then switch to **Selected blocks**. **Full
model** restores the complete diagram; **Reset selected blocks** clears the
selection. Filtering changes only what is shown: all calculations still run,
and selected blocks persist across record, sequence, and update changes. In
Math mode, tracing a variable from a hidden block reveals that block. This also
works in standalone HTML exports.

The Training tab saves your corpus, run name, architecture, and training settings
locally while you work and switch tabs. On startup or reload, the active run
(or otherwise the most recently created run) supplies the corpus, name, model,
and training settings. A local draft is used when there are no saved runs. Numeric controls
stay compact to leave more room for the loss chart. New runs plot both training
loss and validation loss. The training curve averages nearby batches into at most
300 points spanning the whole run; the latest-loss statistic remains the raw
batch loss. Older truncated histories recover earlier measured points from saved
records when available. Validation evaluates up to 16 fixed windows from the
held-out split at the first and final steps and periodically in between (roughly
100 checkpoints for long runs). It does not update weights or consume the training
sampler's random numbers. Older runs without validation measurements say so.
Select a recent run to load its corpus, architecture, training settings, and name alongside its saved progress, curves, and log; use its **Explore**
link to open the model diagram. The current selection is highlighted. Starting training immediately clears the
previous graph and focuses the progress panel. Every run initializes a fresh model;
the same seed and settings deliberately reproduce the same starting curve.
Recent runs include **Delete** for inactive training runs. Deleted files move to
`out/.trash/` for recovery, and the default example stays available. Other entries, including browser-check, can be deleted.

Enable **Corpus heatmap** to shade each corpus position by its cumulative input
usage across the entire training run. Counts scale linearly from cool blue
(zero uses) through green to warm red (the maximum frequency in this run).
Validation text stays unshaded. Click a highlighted position for exact counts (or press Enter / Space when focused). The selected sequence
keeps its border, and the heatmap stays the same when switching training steps,
batch sequences, or the recorded weight update.

New recordings count every batch, including steps omitted from replay, and
store compact constant-count ranges. Existing recordings can recover exact
counts when their saved seed reproduces every recorded batch; otherwise the
heatmap is disabled rather than showing estimated usage. Standalone exports
include the same heatmap.

The browser fetches one recorded step at a time and caches at most three steps.
Switching steps retains the matrix document and camera. The backend currently
reads each recording JSON into memory; recordings are not yet streamed from an
indexed tensor store. Training jobs run in separate Python processes.

## Train from the browser

Open the **Training** tab beside **Explorer**. Paste your corpus, or use the
current corpus as an example. Configure embedding, Q/K/V, and feed-forward
dimensions, layers, heads, context length, optimizer steps, batch size, learning
rate, and recording interval. More settings include weight decay, random seed,
and the training/validation split. Both text splits must exceed the context
length. Training currently uses local CPU PyTorch with dropout fixed at zero
for exact replay.

Click **Start training** to see step progress, current loss, recorded-step count,
and a live loss curve. **Stop training** cancels the active job. The first and
final steps are always recorded, in addition to the chosen interval. Once
export finishes, **Open in explorer** opens the new model's recording. The
corpus, generated config, checkpoint, recording, progress, and logs are saved
in that run's `out/run-…/` directory. Recent runs are available in the Training
tab; the active or latest run is restored when the page reloads.

Training continues when switching tabs or closing the browser while the local
server is running. Stopping the server also stops its training processes.
Standalone HTML exports support exploration; starting training requires the
local Latte server. The CLI and local API remain available, and any completed
run can also be opened with `/?run=RUN_ID`.

## Record an experiment or export HTML

After the launcher has installed dependencies:

```bash
.venv/bin/python latte.py 34324 --record 1000 10 10
```

The five digits set model dimension, shared Q/K/V head dimension, head count,
layer count, and FFN dimension. Six-digit codes additionally separate QK and V
dimensions. Each invocation trains a fresh run; no `--retrain` flag is needed.
The generated config is saved as `out/default/architecture.toml`.

To supply a TOML config instead:

```bash
.venv/bin/python latte.py --config configs/default.toml --record 10 5 2 --output-dir out/experiment
```

This records five steps starting at step 10, spaced two steps apart. Open
`http://127.0.0.1:8000/?run=experiment` in the server version, or open
`out/experiment/latte.html` directly as a standalone export. On Windows use
`.venv\Scripts\python.exe` instead of `.venv/bin/python`.

Each run starts from the configured seed. Reusing an output directory replaces
its files; separate directories preserve experiments. Edit `configs/default.toml`
to change architecture, corpus, batch size, learning rate, or seed. Relative
corpus paths resolve from the Latte project. Replay requires `dropout = 0`.
Recording duration comes from `--record`.

For longer training without matrix recording:

```bash
.venv/bin/python train.py --config configs/default.toml --steps 2000
```

That command writes a checkpoint without an explorer. Replay recordings begin
as new seeded runs; they do not resume that checkpoint.

## Local API

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/runs` | List saved runs and process status |
| `POST /api/runs` | Start a separate training process |
| `GET /api/runs/{id}/status` | Read process status and recent training log |
| `POST /api/runs/{id}/stop` | Stop an active training process |
| `GET /api/runs/{id}/metadata` | Get architecture, corpus, vocabulary, and step summaries |
| `GET /api/runs/{id}/steps/{index}` | Fetch one recorded step, using a zero-based recording index |
| `GET /api/runs/{id}/export` | Download the original standalone HTML |

Example start request:

```bash
curl -X POST http://127.0.0.1:8000/api/runs \
  -H 'Content-Type: application/json' \
  -d '{"config":"configs/default.toml","start":1,"count":10,"stride":1}'
```

The response includes the run ID. Poll its status, then open `/?run=RUN_ID` once
its state is `complete`. Failed jobs expose their error log through the status
endpoint. Training is not yet displayed live inside the replay view. Restarting
the server stops its active training jobs; completed outputs remain on disk.

## Development

The `latte/` package, browser assets, and corpora are independent of littleLLAMA.
The shared forward-calculation code still draws the matrices; React owns replay
state and controls, while D3 owns the prediction table's body and the presentation
adapter's SVG outlines/ribbons. Keyed SVG joins retain nodes during scrolling;
labels and connectors update together before paint. The adapter adds MathML
equations to the existing matrix renderer. Typography and borders are configured
in `appearance.css`.

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m unittest discover -s tests
cd frontend
npm ci
npm test
npm run build
```

To rebuild automatically while editing, run `npm run dev` in `frontend` and
refresh the app. Restart the Python server after changing Python code.

Browser checks compare the standalone and server-rendered views, then exercise
step selection, sequence navigation, weight inspection, applying updates, Math
mode, matrix formulas, and clearing selections:

```bash
cd frontend
npx playwright install chromium
npm run test:browser
```

Set `LATTE_BROWSER_CHANNEL=chrome` to use an installed Chrome instead. The browser
tests require a default export (`.venv/bin/python latte.py`) and start a temporary
local server on port 8017.
To preserve a larger default run while testing the compact UI fixture, create
`out/browser-check` with `latte.py --record 1 2 1 --output-dir out/browser-check`,
then run `LATTE_TEST_RUN=browser-check npm run test:browser` in `frontend`.

## Package a release

After rebuilding the interface:

```bash
.venv/bin/python scripts/package_release.py
```

This creates `releases/Latte.zip` with the source and prebuilt UI. It excludes
virtual environments, Node dependencies, experiments, and Git history. Recipients
extract it and run `python launch.py`; Node.js is unnecessary.

The explorer sidebar uses the available height for the corpus. Selected-token
popups open on click or Enter / Space and show predictions, targets, correctness, and P(target) from the recorded
before-update logits; a separate predictions table is no longer needed.

The block chooser follows the diagram’s bracket hierarchy: layers contain
Attention, heads, MLP, and residual groups. Checking or clearing a group updates
all its descendant blocks; partially selected groups show a mixed checkbox.
Only block leaves are saved as the display selection, so calculations remain
unchanged and selection persists across replay changes.

In **Pick on diagram**, the same group checkboxes appear beside the left
brackets. Click a group checkbox or its bracket label to select or clear all
nested blocks. Diagram and list selections share the same partial-selection
state. Group controls disappear when picking finishes.

The live app shows a loading screen before its data and initial diagram layout
are ready, keeping the underlying export workbook hidden during reloads.

The product title is **One Small Latte**, set in locally bundled Fraunces
SemiBold. Architects Daughter remains the diagram handwriting. Font licenses
are included in `fonts/`; both the live app and standalone exports work offline.

Click a highlighted corpus token to connect it to the corresponding input-token
cell above E. The link uses sequence position, including spaces and repeated
characters. Only one connection is shown. If the input is offscreen or filtered
out, **Go to input** reveals and focuses it. Clicking elsewhere, pressing Escape,
or changing the replay selection clears the connection.
