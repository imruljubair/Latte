# GitHub Pages demo

The Training tab offers three real CPU PyTorch recordings on the bundled story
corpus. The two-layer, two-head baseline runs for 201 steps. The one-layer,
one-head example runs for 121 steps. The longer-context example uses 12 characters
per sequence and runs for 161 steps. Choose a configuration and click **Replay
training** to watch saved loss measurements and checkpoint counts advance.
Custom training runs locally.

Explorer supports checkpoint and sequence selection, before/after-update
comparison, matrix and cell math, operand ribbons, block filtering, zoom, pan,
smooth input/output navigation, corpus connectors, and cumulative corpus usage.

## Build and preview

```sh
npm ci --prefix frontend
npm run build --prefix frontend
python3 -m pip install 'numpy>=1.26'
python3 scripts/build_pages.py
python3 -m http.server 8794 --directory _site
```

Open http://localhost:8794. Every asset uses project-relative paths, including
when hosted under `/Latte/`. The builder verifies checksums and regenerates the
explorer from the current sources. The Python builder needs no PyTorch or checkpoint. Build the shared React UI
with Node.js first. Visitors need no local API or external font/CDN requests. `_site` is generated and ignored.
Recorded tensors are compressed in `demo/data` and published as per-example JSON files.

## Refresh the public recordings

With Latte's local dependencies installed:

```sh
.venv/bin/python scripts/record_demo.py
.venv/bin/python scripts/build_pages.py
```

The recorder reads `demo/config.toml` and the overrides in `demo/presets.json`.
Pass a preset ID such as `single-head` to refresh just that recording. Checkpoints
are generated in temporary directories. Commit the updated public files in
`demo/data`.

## Publish with GitHub Desktop

Open the existing Latte repository in GitHub Desktop. Commit to `main`, then
**Push origin**. The Pages workflow builds and tests the frontend, verifies the
recording, builds `_site`, and deploys it. Under repository Settings → Pages,
select **GitHub Actions** as the source. The first repository publication must
be public for GitHub Pages on a free account.

Expected demo: https://imruljubair.github.io/Latte/
Source: https://github.com/imruljubair/Latte

## Checks

```sh
npm run build --prefix frontend
npm test --prefix frontend
python3 -m unittest discover -s tests -p 'test_pages.py' -v
python3 scripts/build_pages.py
```

The local app does not load `demo/demo.js` and retains the full Training tab.
