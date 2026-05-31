# B300 Cluster Handoff — Qwen3.6-27B Reparse

## Summary

Migrate the Qwen3.6-27B OCR-reparse pipeline from a single-GB10 host to an
8 × B300 cluster. Run 8 independent vLLM replicas (one per GPU), front
them with an nginx round-robin LB on `:8000`, point the existing
`hybrid` parse runtime at the LB, apply the cutover deletion list, and
let the watchdog auto-relaunch the parse pipeline.

**Scope.** This doc is the only thing the new agent needs. Every section
is copy-paste-runnable end-to-end. Decision points are flagged inline.

**Origin state on the GB10 source host (2026-05-30 21:30 UTC+7):**

* Repo: `origin/main` at `ebf08e3`. Working tree clean. All
  qwen3.6 / hybrid surgical / pre-classifier / parsing.md changes are
  committed and pushed.
* Pre-classifier output: `/home/quantm/vllm/qwen3.6-omni/logs/preclassify/{per_doc.parquet,summary.json,pdf_list.txt,run.log}` — must be transferred separately (not in `$HOME/data`).
* Cutover deletion list: `/home/quantm/vllm/qwen3.6-omni/logs/cutover/{delete_list.txt,delete_summary.json}` — same.
* OCR cohort: B' = 168,347 + C = 340,524 + D = 84,870 = **593,741 docs / ~2.6M pages**.
* Stale `.md` already deleted on GB10: 88,335 (B' = 7,237 + C = 101 + D = 80,997). Whatever `$HOME/data` you ship is post-deletion. The cluster does **not** need to re-delete unless the data snapshot was taken before 05:39 UTC on 2026-05-30 — see § 8.

**Throughput target.** GB10 baseline ≈ 85 docs/hr (single GB10 GPU,
Qwen3.6-27B-FP8 + MTP, `--max-num-seqs 8`). Per-replica B300
throughput at BF16 with batch ≥ 64 should land around 200-400 docs/hr
(single GPU). Eight replicas → **~1,600-3,200 docs/hr cluster-wide**.
Full 593,741-doc reparse ETA: **8-15 days wall-clock**, vs. GB10's
~7-12 months. Tune § 5 / § 9 if you see lower numbers.

---

## 0. Hardware & software preflight

```bash
nvidia-smi -L                               # expect 8 × B300
nvidia-smi --query-gpu=driver_version,cuda_version --format=csv,noheader | head -1
docker info | rg -i nvidia                  # nvidia-container-toolkit must be present
df -h $HOME                                 # need ≥ 1 TB free for data + weights + logs
free -h                                     # ≥ 256 GB system RAM is comfortable; ≥ 128 GB minimum
nproc                                       # ≥ 32 cores keeps the pre-classifier reasonable
```

**Required minima:**

| Component | Min | Notes |
|-----------|-----|-------|
| NVIDIA driver  | 575.x | B300 (Blackwell Ultra) needs the recent driver line. |
| CUDA           | 12.8  | Ships with the driver line above. |
| Docker         | 27.x  | Plus `nvidia-container-toolkit ≥ 1.16`. |
| Disk free      | 1 TB  | Data ≈ 600 GB, weights ≈ 60 GB, logs/cache ≈ 100 GB. |
| Open ports     | 8000-8008, plus 22 for ssh | 8000 = LB, 8001-8008 = per-replica. All bound to 127.0.0.1. |

**Stop here if any check fails.** Don't proceed on a half-configured
host — the rest of the doc assumes the preflight passes.

---

## 1. Repo + data sync

The user has already cloned ViLA to `$HOME/ViLA` and symlinked
`$HOME/data → $HOME/ViLA/data`. Verify and pull latest.

```bash
cd $HOME/ViLA
git fetch origin
git checkout main
git pull --ff-only origin main
git log -1 --oneline                         # expect ebf08e3 or newer

ls -la $HOME/ViLA/data                       # confirm symlink
ls $HOME/ViLA/data/congbobanan.toaan.gov.vn/pdf/ | wc -l   # expect ~2.03M
ls $HOME/ViLA/data/congbobanan.toaan.gov.vn/md/  | wc -l   # expect ~2.74M (md + meta = 2 files/doc)
test -f $HOME/ViLA/data/congbobanan.toaan.gov.vn/_watchdog.sh && echo OK
```

If the data transfer is still in progress: wait. Don't start parse
until `pdf/` count is steady and matches the source host.

**Required code anchors** (sanity-check the pull landed the right
commits):

```bash
rg -l "class Qwen36OmniClient"   $HOME/ViLA/packages/parser/qwen3_6_omni.py
rg -n "surgical_pages: bool"     $HOME/ViLA/packages/parser/hybrid.py
rg -n "pypdf\+qwen3.6-27b"       $HOME/ViLA/packages/common/schemas.py
rg -n "hybrid_fallback_runtime"  $HOME/ViLA/packages/datasites/congbobanan/configs/default.yaml
```

All four must return at least one hit. If any are missing, the pull
didn't land — re-check the remote.

---

## 2. Transfer auxiliary artifacts

The pre-classifier outputs and the cutover delete list live **outside
`$HOME/data`** on the GB10 source host. Either rsync them or re-run
on the cluster.

### 2.1 Option A — rsync from GB10 (recommended; saves 30-60 min)

```bash
mkdir -p $HOME/vllm/qwen3.6-omni/logs/{preclassify,cutover}

# From the cluster, pulling from GB10:
rsync -av --progress \
  quantm@<GB10_HOST>:/home/quantm/vllm/qwen3.6-omni/logs/preclassify/ \
  $HOME/vllm/qwen3.6-omni/logs/preclassify/

rsync -av --progress \
  quantm@<GB10_HOST>:/home/quantm/vllm/qwen3.6-omni/logs/cutover/ \
  $HOME/vllm/qwen3.6-omni/logs/cutover/

# Sanity:
ls -la $HOME/vllm/qwen3.6-omni/logs/preclassify/    # per_doc.parquet (60 MB), summary.json, pdf_list.txt
ls -la $HOME/vllm/qwen3.6-omni/logs/cutover/        # delete_list.txt (11 MB, 176k lines), delete_summary.json
wc -l $HOME/vllm/qwen3.6-omni/logs/cutover/delete_list.txt   # expect 176670
```

### 2.2 Option B — re-run pre-classifier on the cluster

If rsync is not feasible, re-run. It is pypdf-only, ~30-60 min on a
32+ core box:

```bash
cd $HOME/ViLA
uv venv
uv sync
uv run python /home/quantm/vllm/qwen3.6-omni/scripts/preclassify_pdfs.py
# (Note: hard-coded paths in the script reference /home/quantm/...
# adjust INPUT_DIR / OUT_DIR at the top of the script if your $HOME
# differs.)
```

Then build the delete list:

```bash
uv run python /home/quantm/vllm/qwen3.6-omni/scripts/build_delete_list.py
```

The expected counts (from the GB10 run, 2026-05-30):

```
counts_by_case: { A: 1381445, B: 53166, B': 168347, C: 340524, D: 84870, ERROR: 4 }
delete_summary: { B': 7237, C: 101, D: 80997, total: 88335 }
```

If your numbers differ by > 5%, stop and investigate before deleting
anything.

---

## 3. Python deps + system tools

```bash
cd $HOME/ViLA
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv missing
source $HOME/.local/bin/env || true
uv venv
uv sync                                            # installs pypdf, pikepdf, docx2txt, pypdfium2, ray, ...

# DOC handler (needed for ~0.03% of the corpus that is .doc):
sudo apt-get install -y antiword catdoc libreoffice

# Sanity:
uv run python -c "import pypdf, pikepdf, docx2txt, pypdfium2, ray; print('ok')"
```

---

## 4. Model weights

User asked for FP16. Qwen/Qwen3.6-27B's published checkpoint is BF16,
which is the precision-equivalent and slightly better dynamic-range
choice on Blackwell. We serve as BF16; it is what the user means
when they say "FP16 unquantized". If you genuinely need IEEE float16
override `--dtype float16` in § 5 — but BF16 is recommended.

```bash
mkdir -p $HOME/vllm/qwen3.6-omni/models
cd $HOME/vllm/qwen3.6-omni/models

# Use the project HF token if private; otherwise omit.
huggingface-cli login                                 # paste token if needed

huggingface-cli download Qwen/Qwen3.6-27B \
  --local-dir Qwen3.6-27B \
  --local-dir-use-symlinks False
```

Expected size on disk: ~54 GB. Verify:

```bash
du -sh $HOME/vllm/qwen3.6-omni/models/Qwen3.6-27B
ls $HOME/vllm/qwen3.6-omni/models/Qwen3.6-27B/*.safetensors | wc -l
```

If you'd rather use the pre-quantized FP8 build (~28 GB, faster
decode on Blackwell, identical Vietnamese OCR quality based on
GB10 production), swap the model name to `Qwen/Qwen3.6-27B-FP8`.
The rest of the doc is the same.

---

## 5. Launch 8 vLLM replicas

One vLLM container per GPU, ports 8001-8008, **bound to 127.0.0.1**
(public 8000 is the nginx LB, § 6).

```bash
WEIGHTS=$HOME/vllm/qwen3.6-omni/models/Qwen3.6-27B
HF_CACHE=$HOME/.cache/huggingface
LOG_DIR=$HOME/vllm/qwen3.6-omni/logs
mkdir -p "$LOG_DIR/replica"

# Stop / clean any prior replicas:
for i in 0 1 2 3 4 5 6 7; do docker rm -f vllm-qwen-$i 2>/dev/null || true; done

for i in 0 1 2 3 4 5 6 7; do
  port=$((8001 + i))
  docker run -d --name="vllm-qwen-$i" \
    --runtime=nvidia \
    --gpus "device=$i" \
    --ipc=host --shm-size=16g \
    -p 127.0.0.1:$port:8000 \
    --restart unless-stopped \
    -v "$WEIGHTS:/model:ro" \
    -v "$HF_CACHE:/root/.cache/huggingface:rw" \
    vllm/vllm-openai:v0.20.0 \
      --model /model \
      --served-model-name qwen3.6-27b \
      --host 0.0.0.0 --port 8000 \
      --trust-remote-code \
      --dtype bfloat16 \
      --max-model-len 32768 \
      --max-num-seqs 64 \
      --gpu-memory-utilization 0.92 \
      --kv-cache-dtype auto \
      --limit-mm-per-prompt '{"image": 1}' \
      --reasoning-parser qwen3 \
      --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
      >/dev/null
done

# Watch them come up (each takes ~60-90s to load weights + warm KV cache):
for i in 0 1 2 3 4 5 6 7; do
  port=$((8001 + i))
  echo -n "replica $i ($port): "
  curl -s http://127.0.0.1:$port/v1/models | rg -o '"id":"[^"]+"' | head -1
done
# Expect: "id":"qwen3.6-27b" on all 8.
```

**Per-replica budget (B300, 288 GB HBM each):**

* BF16 weights: ~54 GB
* `gpu-memory-utilization 0.92` reserves ~265 GB of the 288 GB
* KV cache (`max_model_len 32768`, `max_num_seqs 64`): ~150 GB
* Activations + headroom: ~60 GB

Plenty of headroom; bump `--max-num-seqs` to 96 or 128 if you want
more batch concurrency at the cost of per-request latency.

**Why 8 independent replicas, not TP=8:** Qwen3.6-27B fits trivially
in a single B300. TP=8 would shard a 54 GB model across 8 × 288 GB
GPUs and pay all-reduce on every token — 100% wasted. 8 replicas
gives 8× throughput with zero NCCL overhead.

---

## 6. Front the pool with nginx

```bash
sudo apt-get install -y nginx

sudo tee /etc/nginx/sites-available/qwen-pool >/dev/null <<'NGINX'
upstream qwen_pool {
    least_conn;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
    server 127.0.0.1:8003;
    server 127.0.0.1:8004;
    server 127.0.0.1:8005;
    server 127.0.0.1:8006;
    server 127.0.0.1:8007;
    server 127.0.0.1:8008;
    keepalive 64;
}

server {
    listen 127.0.0.1:8000;
    server_name localhost;

    proxy_http_version 1.1;
    proxy_set_header   Connection "";
    proxy_read_timeout 600s;
    proxy_send_timeout 600s;
    proxy_buffering    off;
    client_max_body_size 64m;

    location / {
        proxy_pass http://qwen_pool;
    }
}
NGINX

sudo ln -sf /etc/nginx/sites-available/qwen-pool /etc/nginx/sites-enabled/qwen-pool
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx

# Sanity:
curl -s http://127.0.0.1:8000/v1/models | rg qwen3.6-27b
```

**Why `least_conn` and not `round_robin`:** OCR requests vary 5×
in latency by page complexity; least-conn naturally drains a
free replica before piling another request onto a busy one.
`keepalive 64` reuses upstream sockets so you don't pay TCP
handshake on every page.

---

## 7. Smoke test the pool

Run a one-page Vietnamese-OCR sanity check before kicking off the
full pipeline:

```bash
cd $HOME/ViLA
uv run python - <<'PY'
import os, base64
from packages.parser.qwen3_6_omni import Qwen36OmniClient

os.environ.setdefault("QWEN3_6_OMNI_BASE_URL", "http://127.0.0.1:8000/v1")
os.environ.setdefault("QWEN3_6_OMNI_MODEL",    "qwen3.6-27b")

client = Qwen36OmniClient()
sample = next(iter([
    p for p in __import__("os").scandir(
        "data/congbobanan.toaan.gov.vn/pdf"
    ) if p.name.endswith(".pdf")
]))
with open(sample.path, "rb") as fh:
    out = client.parse(fh.read())
md = out["markdown"]
print(f"chars={len(md)}  pages={len(out.get('pages', []))}")
print(md[:600])
PY
```

Expected: a few hundred to a few thousand chars of Vietnamese
text with full diacritics (`đ`, `ấ`, `ợ`, `ầ`, ...). No Chinese
drift, no `<think>` preamble. If you see Chinese, the
`enable_thinking` flag in the client isn't being honored — check
`packages/parser/qwen3_6_omni.py` for the `chat_template_kwargs`
extra_body.

Run the test 8 times in a row and watch `nvidia-smi -l 2` —
utilization should rotate across all 8 GPUs (least-conn balancing).

---

## 8. Apply cutover deletions (idempotent)

The 88,335 stale `.md` (B' + C + D from old nemotron-parse runs)
were already deleted on GB10 at 05:39 UTC 2026-05-30. The data
snapshot you transferred reflects that. Skip this step **unless**
your snapshot pre-dates the deletion.

To verify (the safe path):

```bash
# Sample 100 random doc_ids from delete_list.txt and check none of them
# still have an .md sidecar. If all are absent, the snapshot is post-cutover.
shuf -n 100 $HOME/vllm/qwen3.6-omni/logs/cutover/delete_list.txt \
  | while read p; do test -e "$p" && echo "STILL PRESENT: $p"; done | head
# Expect: empty output. Any "STILL PRESENT" lines mean re-run the deletion.
```

If you must re-delete:

```bash
# Idempotent — `rm -f` skips missing files silently.
xargs -a $HOME/vllm/qwen3.6-omni/logs/cutover/delete_list.txt rm -f
# Sanity: md/ count should drop by ~88,335 docs (~176,670 files).
ls $HOME/ViLA/data/congbobanan.toaan.gov.vn/md/ | wc -l
```

---

## 9. Tune parse pipeline for an 8-replica pool

The default `stage_overrides.parse_files_per_partition: 8` was
sized for the GB10 single-GPU bottleneck. Bump for the B300 cluster
so the pool stays saturated.

Edit `$HOME/ViLA/packages/datasites/congbobanan/configs/default.yaml`
(or pass `--override` flags at launch — preferred, no commit needed):

| Override | Default | B300 cluster |
|----------|---------|--------------|
| `stage_overrides.parse_files_per_partition` | `8` | `64` |
| `parser.num_workers` (Ray actor count for `PdfParseStage`) | `15` | `64-96` |
| `parser.qwen3_6_omni_max_concurrency` (per-actor inflight) | `1` | `1` (let nginx handle queuing) |

Aim for **≥ 8 inflight requests per replica** (so 64+ across the
pool). Watch `nvidia-smi -l 5` — sustained ~80-90% GPU util on
all 8 means you're balanced. Below 50% means add actors;
above 95% with rising p99 latency means back off.

---

## 10. Launch watchdog + parse

Use the existing watchdog (it already speaks the env-var contract
for `qwen3_6_omni`):

```bash
cd $HOME/ViLA

setsid nohup env \
  PARSER_RUNTIME=hybrid \
  HYBRID_FALLBACK_RUNTIME=qwen3_6_omni \
  QWEN3_6_OMNI_BASE_URL=http://127.0.0.1:8000/v1 \
  QWEN3_6_OMNI_MODEL=qwen3.6-27b \
  bash data/congbobanan.toaan.gov.vn/_watchdog.sh \
  >/dev/null 2>&1 < /dev/null &
disown

sleep 5
cat $HOME/ViLA/data/congbobanan.toaan.gov.vn/_watchdog.pid
pgrep -af _watchdog.sh
```

Wait one watchdog tick (60s) and confirm parse came up:

```bash
sleep 65
tail -20 $HOME/ViLA/data/congbobanan.toaan.gov.vn/logs/latest-watchdog.log
pgrep -af 'packages.datasites.congbobanan --pipeline parse'
```

Expected log lines:

```
ALERT: parse pid=none not running; <md>/<pdf> parsed; relaunching
launching parse -> .../parse-<stamp>.log (PARSER_RUNTIME=hybrid)
    parse relaunched pid=<NEW_PID>
```

To **override the parse_files_per_partition without touching YAML**,
edit `_watchdog.sh::launch_parse` once on the cluster to add a
`--override stage_overrides.parse_files_per_partition=64` flag
(line ~152 in the file). Or commit an env-var read for it — your
call.

---

## 11. Monitor

**Watchdog status (10-min cadence):**

```bash
tail -f $HOME/ViLA/data/congbobanan.toaan.gov.vn/logs/latest-watchdog.log
# STATUS lines show: parse_alive, tmp%, md count
```

**vLLM pool health:**

```bash
for i in 0 1 2 3 4 5 6 7; do
  port=$((8001+i))
  echo "=== vllm-qwen-$i ($port) ==="
  docker logs --tail 5 vllm-qwen-$i 2>&1 | rg -i 'error|warn|throughput' | tail -3
done

# Throughput / queue depth:
for i in 0 1 2 3 4 5 6 7; do
  port=$((8001+i))
  curl -s http://127.0.0.1:$port/metrics 2>/dev/null | \
    rg 'vllm:num_requests_(running|waiting)|vllm:generation_tokens_total' | head -3
done
```

**GPU utilization:**

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv -l 5
```

**Throughput (real-time docs/hr):**

```bash
watch -n 60 'echo "$(date -u +%H:%M:%S)  $(ls $HOME/ViLA/data/congbobanan.toaan.gov.vn/md/ | wc -l) files"'
# Each doc emits 2 files (.md + .meta.json), so divide by 2 for doc count.
# Subtract two consecutive snapshots to get docs/hr.
```

**Parse pipeline log (Ray/Xenna stage table):**

```bash
tail -f $HOME/ViLA/data/congbobanan.toaan.gov.vn/logs/latest-parse.log
# Look for: PdfParseStage Speed (tasks/actor/s), Slots Num Used, Tasks Completed
```

---

## 12. Acceptance criteria — when to call it healthy

After ~30 min of steady-state run, **all** of the following should
hold:

| Signal | Healthy range |
|--------|---------------|
| `nvidia-smi` GPU util (8 GPUs) | 70-90% sustained |
| GPU memory used (8 GPUs) | ~250 GB / 288 GB (consistent with `gpu-memory-utilization 0.92`) |
| `vllm:num_requests_running` per replica | 8-32 |
| `vllm:num_requests_waiting` per replica | 0-16 spikes, no monotone climb |
| `PdfParseStage` actors running | ≥ 80% of `--num-workers` |
| docs/hr cluster-wide | ≥ 1500 |
| Sample 5 fresh `.md` (Vietnamese diacritics, no Chinese, ≥ 200 chars) | all pass |

If any of those fail, see § 13.

---

## 13. Troubleshooting

### Replica won't start (OOM, model load fails)

```bash
docker logs vllm-qwen-0 2>&1 | tail -50
```

Common causes:

* `gpu-memory-utilization 0.92` too aggressive on a host with other
  processes on the GPU. Drop to 0.85 and restart that replica.
* Driver / CUDA version mismatch → `nvidia-smi` and check.
* Stale weight cache. Remove `$WEIGHTS` and re-download.

### Parse stalls (PdfParseStage Speed → 0)

```bash
# Check vLLM endpoints aren't 502'ing:
for i in 0 1 2 3 4 5 6 7; do curl -sS -o /dev/null -w "%{http_code}\n" http://127.0.0.1:$((8001+i))/v1/models; done

# nginx upstream errors:
sudo tail -50 /var/log/nginx/error.log

# Actor-side errors:
rg -i "error|exception" $HOME/ViLA/data/congbobanan.toaan.gov.vn/logs/latest-parse.log | tail
```

If a replica is wedged, just restart it:

```bash
docker restart vllm-qwen-3
```

The watchdog will keep parse alive while the pool degrades to 7
replicas; nginx routes around the dead upstream.

### Vietnamese drift (Chinese characters in output)

Means `enable_thinking=False` did not propagate. Check the
`extra_body.chat_template_kwargs` block in
`packages/parser/qwen3_6_omni.py`. The fix shipped on GB10; if
your repo is at `ebf08e3` or newer it should already be correct.

### `/tmp` filling up (Ray spill)

The watchdog's `TMP_DANGER_PCT=93` will SIGTERM parse and refuse to
relaunch until disk recovers. Either raise `--tmp-dir` to a bigger
volume by editing the parse launch line in `_watchdog.sh`, or
prune old Ray spill:

```bash
sudo rm -rf /tmp/ray/session_*  # only when parse is stopped
```

### Rollback to GB10 hosting

If the cluster experiment fails:

1. `kill $(cat $HOME/ViLA/data/congbobanan.toaan.gov.vn/_watchdog.pid)`
2. `for i in 0 1 2 3 4 5 6 7; do docker rm -f vllm-qwen-$i; done`
3. Resume parse on the GB10 host (its watchdog/parse PIDs are 962316/962589 as of 2026-05-30 21:30 UTC+7; restart manually if dead).
4. The cluster's fresh `.md` files are valid output — rsync them back to GB10's `data/congbobanan.toaan.gov.vn/md/` and the GB10 watchdog will skip them via `SkipExistingMarkdownFilter`.

---

## 14. Post-completion — when the OCR cohort finishes

Indicator: `md/` count steady at ~2.03M docs, watchdog STATUS line
shows `pdf_n == md_n`, parse PID exits naturally.

1. Run a final coverage report:

   ```bash
   cd $HOME/ViLA
   uv run python /home/quantm/vllm/qwen3.6-omni/scripts/preclassify_pdfs.py \
     # optional: re-run with the now-complete .md set to confirm
     # zero remaining Case-B'/C/D rows without sidecars
   ```

2. Tear down the pool to free GPUs:

   ```bash
   for i in 0 1 2 3 4 5 6 7; do docker rm -f vllm-qwen-$i; done
   sudo systemctl stop nginx
   ```

3. rsync the fresh `md/` back to GB10 (or wherever the canonical
   corpus lives). `.meta.json` sidecars include
   `parser_model: pypdf+qwen3.6-27b` and `parsed_at` so downstream
   stages can detect the new run.

4. Hand off to the downstream extractor / embedder pipeline.

---

## Appendix A — quick reference

| Resource | Path |
|----------|------|
| Repo | `$HOME/ViLA` (commit ≥ `ebf08e3`) |
| Data | `$HOME/data → $HOME/ViLA/data` |
| Pre-classifier output | `$HOME/vllm/qwen3.6-omni/logs/preclassify/` |
| Cutover delete list | `$HOME/vllm/qwen3.6-omni/logs/cutover/delete_list.txt` |
| Model weights | `$HOME/vllm/qwen3.6-omni/models/Qwen3.6-27B/` |
| vLLM containers | `vllm-qwen-{0..7}` on ports 8001-8008 |
| nginx LB | `127.0.0.1:8000` |
| Parse watchdog log | `data/congbobanan.toaan.gov.vn/logs/latest-watchdog.log` |
| Parse pipeline log | `data/congbobanan.toaan.gov.vn/logs/latest-parse.log` |

| Env var | Value |
|---------|-------|
| `PARSER_RUNTIME` | `hybrid` |
| `HYBRID_FALLBACK_RUNTIME` | `qwen3_6_omni` |
| `QWEN3_6_OMNI_BASE_URL` | `http://127.0.0.1:8000/v1` |
| `QWEN3_6_OMNI_MODEL` | `qwen3.6-27b` |

| Counts (origin, 2026-05-30) |  |
|---|---|
| Total docs | 2,028,356 |
| Case A (pypdf-clean) | 1,381,445 |
| Case B (cmap-rescued) | 53,166 |
| Case B' (corrupt-font OCR) | 168,347 |
| Case C (image-only OCR) | 340,524 |
| Case D (mixed surgical OCR) | 84,870 |
| **OCR cohort total** | **593,741** |
| Stale `.md` deleted on GB10 | 88,335 |

---

*End of handoff. Ping the GB10 agent with any open questions; the
full conversational context is in
`/home/quantm/.cursor/projects/home-quantm/agent-transcripts/38fedce8-4382-4c7e-87fd-6ff7481a23b1/`.*
