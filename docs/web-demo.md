# GitHub Pages demo

The online edition explores a real, seeded CPU PyTorch training recording.
The public example uses the bundled story corpus, two layers, two heads, eight
characters per sequence, and 201 AdamW updates. Checkpoints at steps 1, 41, 81,
121, 161, and 201 contain the actual weights, gradients, and batches.

Explorer supports checkpoint and sequence selection, before/after-update
comparison, matrix and cell math, operand ribbons, block filtering, zoom, pan,
smooth input/output navigation, corpus connectors, and cumulative corpus usage.
Training shows the saved training and validation loss curves, exact settings,
and the same corpus. New training and custom corpora require the local app;
the demo links to its setup guide. No browser training or backend is simulated.

## Build and preview

```sh
python3 -m pip install 'numpy>=1.26'
python3 scripts/build_pages.py
python3 -m http.server 8794 --directory _site
```

Open http://localhost:8794. Every asset uses project-relative paths, including
when hosted under `/Latte/`. The builder verifies checksums and regenerates the
explorer from the current sources. It needs no PyTorch, Node.js, local API,
checkpoint, or external font/CDN requests. `_site` is generated and ignored.
Recorded tensors are compressed in `demo/data` and embedded in the built page.

## Refresh the public recording

With Latte's local dependencies installed:

```sh
.venv/bin/python scripts/record_demo.py
.venv/bin/python scripts/build_pages.py
```

This uses `demo/config.toml` and writes only the dedicated `out/pages-demo`
folder. It never exports other local runs. Commit the updated public files in
`demo/data`; checkpoints and private run folders remain ignored.

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
