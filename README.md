# Qwen3.8-27B (NVFP4) on SGLang — DSpark speculative decoding

Run the 27B model on a single **RTX 5090 (Blackwell, 32 GB)** at high decode
speed using SGLang + an NVFP4 checkpoint + a DSpark drafter. One self-contained
repo: prebuilt image, one `setup` command, two launch presets, and an optional
monitoring dashboard.

```
./setup.sh            # pull image + download weights (target + drafter)
./run-sglang-godspeed.sh start
```

That's it. Open `http://localhost:8040/v1` (OpenAI-compatible) or
`/anthropic`.

---

## Key facts

| | |
|---|---|
| Model | Qwen3.8-27B, NVFP4 (weights) + lm_head in NVFP4 + FP8 KV cache |
| Draft | DSpark drafter (separate ~1.4 GB model, speculative decoding) |
| Engine | SGLang (`lmsysorg/sglang:qwen38-27b`, prebuilt, no JIT) |
| GPU | 1× RTX 5090 (32 GB, sm_120 / Blackwell) |
| Target speed | ~180 tok/s (godspeed) / ~150–200 tok/s (vision) |
| Context | ~236k (text-only) / ~150k (vision) |
| Ports | API 8040 · gateway 8041 · Grafana 8042 · Prometheus 127.0.0.1:9091 |

The speed is the DSpark drafter: it proposes a block of tokens the big model
verifies in one step. The higher the accepted-draft rate, the faster the decode.
This repo exposes those numbers live in the Grafana dashboard.

---

## Feature overview

| | |
|---|---|
| ⚡ Inference | Prebuilt SGLang image — boots with **no JIT/compile**; NVFP4 weights + FP8 KV cache, Blackwell SM120 FlashInfer FP4 GEMM |
| 🎲 Speculative decoding | DSpark drafter proposes a block of draft tokens; the 27B model **verifies the whole block in one step** (block 7 godspeed / 5 vision) |
| 👁️ Two presets | **godspeed** (text-only, 236k ctx) ⇄ **vision** (tower on, 150k ctx) — swappable on one GPU |
| 📊 Grafana | Auto-provisioned dashboard: dense KPI tiles + a live **DSpark win-rate** section. `make_dashboard.py` is the source; the JSON is build output |
| 📈 Prometheus | 30-day retention, scrapes SGLang + DCGM — the dashboard's own data source (`127.0.0.1:9091`, local-only) |
| 🎛️ GPU telemetry | DCGM exporter (util / mem / temp / power) alongside `sglang:` engine metrics; every tile aggregated (`sum()`/`max()`) to a single value |
| 🔐 Gateway | Caddy **Basic-auth** on 8041 → OpenAI/Anthropic endpoints with bearer `API_KEY`; auth-off by default |
| 🐳 GPU isolation | Rootless **Podman + NVIDIA CDI** (`--device nvidia.com/gpu=all`); presets free the GPU before boot |
| 🔁 vLLM switch | `VLLM_COMPOSE_DIR` cleanly `down`s a competing vLLM compose (`restart: always`-safe GPU handover) |
| 🏷️ Stable API id | `qwen3.8-27b-nvfp4` via `--served-model-name` (overridable in `.env`), so `/v1/models` never leaks the `/model` path |
| 🧠 Thinking control | Qwen3.8 thinking/reasoning knobs via SGLang `--default-chat-template-kwargs`: `reasoning_effort` (xhigh/medium/low), `enable_thinking`, `preserve_thinking` — set in `.env` as `DEFAULT_CHAT_TEMPLATE_KWARGS` |
| 📦 Self-contained | One `./setup.sh` (image + target + drafter weights → `./models`); share across stacks via `MODELS_ROOT` |

---

## Requirements

- **Linux x86_64**, one RTX 5090, NVIDIA driver ≥ 580 (Blackwell).
- **Podman** (rootless) + the NVIDIA **CDI** device plugin — so `podman run
  --device nvidia.com/gpu=all` works. (Equivalent Docker + nvidia-container-toolkit
  also works; the scripts call `podman`, add a shim if you use Docker.)
- **podman-compose** for the monitoring stack.
- **Python `hf` CLI** (huggingface_hub) for weight downloads.
- ~41 GB for the image + ~20 GB for weights.

---

## 1 · Set up

```bash
git clone <this-repo> sglang && cd sglang
cp .env.example .env            # defaults are fine to start
./setup.sh                      # pulls the image, downloads target + drafter
```

Weights go to `./models/` by default (self-contained). To share one store across
multiple stacks, set `MODELS_ROOT` in `.env` and `setup.sh` symlinks into it.

## 2 · Launch

```bash
./run-sglang-godspeed.sh start      # text-only, max ctx, fastest
# or
./run-sglang-vision.sh start        # vision ON, lower ctx

./run-sglang-godspeed.sh status     # wait until "HTTP 200 (ready)"
```

Both presets free the GPU first (stop any other container using it). Add
`KEEP_GPU=1` to skip that. Swap presets by `stop`-ing one, `start`-ing the other
— you only have one GPU, so they're **swappable, not simultaneous**.

## 3 · Call it

```bash
curl -s http://localhost:8040/v1/models
curl -s http://localhost:8040/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b-nvfp4","messages":[{"role":"user","content":"hello"}]}'
```

The served id is `qwen3.8-27b-nvfp4` (both presets pass
`--served-model-name`, so `/v1/models` no longer leaks the container path
`/model`). Override it per-stack via `SERVED_MODEL_NAME` in `.env`.

---

## The two presets

| | godspeed (text-only) | vision |
|---|---|---|
| Vision tower | off (`--language-only`) | **on** |
| `--context-length` | 237568 | 150000 |
| `--mem-fraction-static` | 0.985 | 0.88 |
| `--speculative-dspark-block-size` | 7 | 5 |
| Expected decode | ~180–260 tok/s | ~150–200 tok/s |
| VRAM | ~31 GB | ~30 GB |

Shared flags: `--kv-cache-dtype fp8_e4m3`, `--attention-backend flashinfer`
(SM120), mamba linear-attention levers (`--mamba-ssm-dtype bfloat16`,
`--mamba-radix-cache-strategy extra_buffer_lazy`), `--max-running-requests 1`.

**Why text-only is faster** (weights are identical): the gap is DSpark
`block-size` 7 vs 5 — bigger accepted draft blocks per verify step — plus 98.5%
VRAM. Lowering ctx is just the cost of fitting the vision tower, not a speed cause.

---

## Thinking / reasoning (Qwen3.8)

Qwen3.8-27B is a thinking model. The chat template exposes three knobs, all
passed to the server as SGLang's `--default-chat-template-kwargs` (a JSON
object applied to every request; a per-request `chat_template_kwargs` in the
API call still overrides the server default):

| knob | values | default | effect |
|---|---|---|---|
| `reasoning_effort` | `xhigh` (default) / `medium` / `low` | `xhigh` | how deeply the model reasons before answering. `medium` balances accuracy vs speed; `low` is fastest/cheapest. (The template rejects any other string; "none"/"off" = set `enable_thinking=false`.) |
| `enable_thinking` | `true` / `false` | `true` | turn the thinking trace on/off entirely (off = no `reasoning_content`, 0 reasoning tokens). |
| `preserve_thinking` | `true` / `false` | `true` | keep the previous turn's thinking trace in the prompt (costs tokens, helps continued conversations). |

**Where it's set:** the launch scripts build `--default-chat-template-kwargs
"$DEFAULT_CHAT_TEMPLATE_KWARGS"`. That variable is `.env`-overridable and
defaults to `{"reasoning_effort":"xhigh"}` when unset (behavior-preserving).

To run medium by default, add to `.env`:

```bash
DEFAULT_CHAT_TEMPLATE_KWARGS='{"reasoning_effort":"medium"}'
```

Other examples:

```bash
# thinking off entirely (cheapest, no reasoning trace)
DEFAULT_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'
# fast + keep prior-turn thinking for ongoing chats
DEFAULT_CHAT_TEMPLATE_KWARGS='{"reasoning_effort":"low","preserve_thinking":true}'
```

Or leave the server default alone and set it per-request (no restart):

```bash
curl -s http://localhost:8040/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-nvfp4","messages":[{"role":"user","content":"hi"}],
       "chat_template_kwargs":{"reasoning_effort":"medium"}}'
```

---

## Monitoring (optional)

Prometheus + Grafana + Caddy + a GPU exporter, all isolated (own ports/network/
volumes). A dedicated **DSpark section** shows live accepted-draft rate.

```bash
./monitor.sh up           # start the monitoring stack
./monitor.sh status        # target health (sglang target is up only while a server runs)
./monitor.sh dashboard     # print the dashboard URL + login hints
```

Dashboard: `http://localhost:8042` → folder *sglang* → "SGLang — Qwen3.8-27B
(DSpark, RTX5090)". Watch `spec_accept_length` / `spec_accept_rate` to see the
drafter win-rate that drives your throughput.

### Live proof (measured via the dashboard's own data source)

Every value below was read through the same Prometheus (port 9091) the
Grafana panels query, on 2026-08-20 20:11 CEST with `sglang-qwen38` up
33 min on the godspeed preset, idle after a few smoke-test requests:

| KPI (Grafana tile) | Query the tile runs | Measured |
|---|---|---|
| Peak generation rate (1h) | `max(sglang:gen_throughput)` | **229.6 tok/s** |
| Drafts accepted / step | `sum(sglang:spec_accept_length)` | 3.9 (peak 5.65 in 1h) |
| Accept rate (acc/proposed) | `sum(sglang:spec_accept_rate)` | 0.414 (peak 0.664 in 1h) |
| Requests (total) | `sum(sglang:num_requests_total)` | 108 |
| Input / generated tokens | `sum(sglang:prompt_tokens_total)` / `…generation_tokens_total` | 5,385,090 / 202,159 |
| TTFT p50 / p95 (1h) | `histogram_quantile(…, sglang:time_to_first_token_seconds_bucket[1h])` | 0.41 s / 1.78 s |
| Startup time (total) | `sum(sglang:startup_time_seconds)` | 77.5 s (incl. 11.4 s CUDA-graph build) |
| Weight VRAM | `max(sglang:weight_memory_usage_gb)` | 16.7 GB |
| KV cache VRAM | `max(sglang:kv_cache_memory_usage_gb)` | 5.0 GB |
| GPU util / temp / power | DCGM `max(…)` | 97% (peak 100%) / 74 °C / 432 W |
| GPU memory | DCGM `max(…)` | 31,212 / 32,607 MiB (95.9%) |

Note the `sum()` / `max()` wrappers: this SGLang build exports several
metrics with multiple label sets (`is_streaming`, `mode`, `phase` — and one
per historical `model_name`), so the dashboard always aggregates before
displaying; a raw gauge would otherwise stack one value per series.

---

## Configuration (`.env`)

Everything is overridable in `.env` — no user-specific paths or usernames are
hardcoded. Common knobs: `SGGLANG_IMAGE`, `MODEL_REPO`/`DRAFTER_REPO`,
`MODEL_SUBDIR`/`DRAFTER_SUBDIR`, `HOST_PORT`, `API_KEY`, `MODELS_ROOT`,
`SERVED_MODEL_NAME` (id exposed at `/v1/models`; default
`qwen3.8-27b-nvfp4`), `HF_HUB_ACCESS_TOKEN`, `METRICS_HASH` (gateway auth).
See `.env.example`.

---

## Troubleshooting / verify before you trust

- **Speed numbers**: ~260 tok/s is community-reported; the model card verifies
  ~180. Re-measure after boot (`status` + a timed decode).
- **OOM at boot** (vision preset): lower `--mem-fraction-static` to 0.80 and
  `--context-length` to ~120000.
- **DSpark vs MTP**: DSpark is the faster drop-in on this hybrid
  Gated-DeltaNet model (the in-checkpoint MTP head is the older K=1 path). No
  vision penalty with DSpark.
- **Metrics target down**: normal until a SGLang server is running; it flips up
  after `./run-sglang-*.sh start` reaches ready.

## Layout

```
sglang/
├── .env.example               # copy to .env (defaults work)
├── setup.sh                   # pull image + download target + drafter
├── run-sglang-godspeed.sh     # text-only preset  (start|stop|logs|status)
├── run-sglang-vision.sh       # vision preset     (start|stop|logs|status)
├── monitor.sh                 # monitoring up|down|status|dashboard
├── docker-compose.yml         # caddy/prometheus/grafana/dcgm (isolated)
├── caddy/Caddyfile            # Basic-auth gateway on 8041
├── prometheus/prometheus.yml  # scrapes the SGLang server + GPU exporter
├── grafana/                   # auto-provisioned datasource + dashboard
│   ├── dashboards/sglang-qwen38-27b.json   # BUILD OUTPUT — do not hand-edit
│   └── provisioning/          # dashboard file-provider (syncs ~30 s) + datasource
├── make_dashboard.py          # dashboard generator (run: python3 make_dashboard.py)
├── dcgm-exporter/            # NVML -> Prometheus GPU exporter
└── models/                    # weights (created by setup.sh; gitignored)
```
