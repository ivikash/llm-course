# Course Website

Static, no build step. Serve and open:

```bash
cd ~/workspace/llm-course
python3 -m http.server 8000
# then visit http://localhost:8000/site/
```

A local server is required (browsers block `fetch()` from `file://`).

## Regenerate the lesson index

After adding or renaming `.md` files, run:

```bash
bash site/build_manifest.sh
```

That updates `site/files.json`, which the sidebar reads at load time.
