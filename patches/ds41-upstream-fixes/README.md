# Two fixes for their tree, found by running it

Both were found bringing `garnermccloud/dgx-station-recipes` up on two DGX Stations with
driver 615.71.09. They are ours to carry until they land upstream; each is a one-line change
and neither alters an answer.

## 1. `ds41:87` — the driver check is inverted

```diff
-    if [ -n "$drv" ] && version_ge 595.58.03 "$drv"; then
+    if [ -n "$drv" ] && version_ge "$drv" 595.58.03; then
```

`version_ge()` (`ds41:67`) is true when its **first** argument is the larger:

```sh
version_ge() { [ "$(printf '%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }
```

so calling it with the floor first makes every driver **newer** than 595.58.03 fail install:

```
595.58.03  → PASS        (equal, by accident)
580.65.06  → PASS        (older, wrong)
610.43.02  → FAIL        (newer, wrong)
615.71.09  → FAIL        (newer, wrong)
```

`ds41 install` stops with `preflight: FAIL (2 to fix, 0 warnings)`.

## 2. `serve/serve.sh:335` — no way to serve without an API key

Their launcher always writes a key into the container's env file, so vLLM always requires a
bearer token. If the lane is behind a router that already authenticates clients, the key is
one more secret to carry for no gain. vLLM enables auth only for a non-empty `VLLM_API_KEY`,
so an empty value is the unauthenticated lane:

```diff
-  chmod 600 "$envf"; echo "VLLM_API_KEY=$(cat "$DS41_API_KEY_FILE")" > "$envf"
+  chmod 600 "$envf"
+  if [ "${DS41_NO_KEY:-0}" = "1" ]; then echo "VLLM_API_KEY=" > "$envf"; else echo "VLLM_API_KEY=$(cat "$DS41_API_KEY_FILE")" > "$envf"; fi
```

Their `preflight.sh` still checks the key file's mode, so the file keeps existing; it is
simply not enforced. Verified: `200` with no header, `200` with a stale key.

## Not reported here

- Their pinned tune file `fastpath/hook/fi-tune/fiset-w6pf-post1-c7483c4a.json` is missing two
  `(K, N)` rows — `(1280, 4096)` has no M=512 and `(5120, 1536)` has no M=3584 — so a prefill
  chunk of 257–512 or 3073–3584 rows logs `No tuned config covers … falling back to
  CuteDSL`. That is a hole in their file, not a patch: adding entries changes the pinned
  artefact, so it belongs upstream.
- `preflight.sh boot` greps the *live* boot log for that string, so a healthy serving lane
  fails its own gate the moment such a prompt arrives, and `verify.sh` then stops before its
  functional stages.
