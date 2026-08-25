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
| Target speed | up to ~323 tok/s measured burst (godspeed @ `medium` default) · ~150–200 tok/s (vision) |
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
| 🐳 GPU isolation | Rootless **Podman + NVIDIA CDI** (`--device nvidia.com/gpu=all`); boot does a read-only VRAM check — it never stops other services |
| 🏷️ Stable API id | `qwen3.8-27b-nvfp4` via `--served-model-name` (overridable in `.env`), so `/v1/models` never leaks the `/model` path |
| 🧠 Thinking control | Qwen3.8 thinking/reasoning knobs via SGLang `--default-chat-template-kwargs`: `reasoning_effort` (medium default / xhigh / low), `enable_thinking`, `preserve_thinking` — set in `.env` as `DEFAULT_CHAT_TEMPLATE_KWARGS` |
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

Start does a read-only VRAM check and never stops other containers or services —
if the 5090 is busy (e.g. a vLLM stack), free it manually first. Swap presets by
`stop`-ing one, `start`-ing the other — you only have one GPU, so they're
**swappable, not simultaneous**.

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
| `--mem-fraction-static` | 0.90 | 0.82 |
| `--max-mamba-cache-size` | 8 | 8 |
| `--speculative-dspark-block-size` | 7 | 5 |
| Drafter | NVFP4 (~1.4 GB, default) · BF16 optional (`DRAFT_MODEL_QUANTIZATION=unquant`) | same |
| Expected decode | ~180 baseline, up to **~323 tok/s** measured burst @ `medium` (godspeed) | ~150–200 tok/s (vision) |
| VRAM | ~31 GB | ~30 GB |

Shared flags: `--kv-cache-dtype fp8_e4m3`, `--attention-backend flashinfer`
(SM120), mamba linear-attention levers (`--mamba-ssm-dtype bfloat16`,
`--mamba-radix-cache-strategy extra_buffer_lazy`), `--max-running-requests 1`.
Sizing and drafter knobs are `.env`-overridable: `MAX_MAMBA_CACHE_SIZE`
(default 8), `MEM_FRACTION_STATIC`, `DSPARK_BLOCK_SIZE`,
`DRAFT_MODEL_QUANTIZATION` (see `.env.example`).

**Why text-only is faster** (weights are identical): the gap is DSpark
`block-size` 7 vs 5 — bigger accepted draft blocks per verify step — plus the
higher `--mem-fraction-static` (0.90 vs 0.82). Lowering ctx is just the cost
of fitting the vision tower, not a speed cause.

### Drafter dtype (NVFP4 vs BF16)

The default drafter is the ~1.4 GB **NVFP4** build. Community testing on the
5090 (single-request, thinking ON) found the ~2.5 GB **BF16** RadixArk DSpark
build drafts noticeably better: accept length ~4.0 (~171 tok/s) vs ~1.7
(~93 tok/s) — so the BF16 drafter is the faster choice when you want maximum
decode speed and can spare ~1.1 GB of VRAM:

```bash
./setup.sh bf16-drafter        # downloads RadixArk/Qwen3.8-27B-DSpark
# in .env:
DRAFTER_SUBDIR=Qwen3.8-27B-DSpark-BF16
DRAFT_MODEL_QUANTIZATION=unquant
```

Restart the preset afterwards (`./run-sglang-godspeed.sh start`). The image's
`--speculative-draft-model-quantization` accepts `unquant` (BF16),
`modelopt_fp4` (NVFP4, the default), and the usual SGLang quant list.

### Mamba cache sizing (multi-session use)

The model is a hybrid Gated-DeltaNet, so each cached request path holds a
mamba/SSM state slot (`--max-mamba-cache-size`). At the old default of 5
slots, long multi-turn sessions with cached-prefix accumulation (~180K ctx)
fill 4/5 slots and trigger eviction-recompute stalls (3–14 tok/s). The stack
now ships **8 slots** by default (cost: ~5K of the token pool, 210K → ~205K),
which moves the incident threshold to 4/8. `MAX_MAMBA_CACHE_SIZE` is
`.env`-overridable — raise it further if you run very long concurrent
sessions, or drop it back to 5 if you want the maximum token pool for
single-shot long-context work.

### Swapping the target model (safetensors)

The target model and the drafter are two **independent** mounts, wired by two
independent `.env` knob pairs:

| Role | `.env` keys | mount | server flag |
|---|---|---|---|
| Target (main model) | `MODEL_REPO` / `MODEL_SUBDIR` | `/model` | `--model-path /model` |
| DSpark drafter | `DRAFTER_REPO` / `DRAFTER_SUBDIR` | `/model_dspark` | `--speculative-draft-model-path /model_dspark` |

So the target is not tied to the drafter (or vice versa):

```bash
# e.g. another quant of the same base (FP8, FP16, a re-quant or re-merge),
# or any other Qwen3.8-27B safetensors build. `setup.sh` downloads it and
# symlinks it into ./models/.
MODEL_REPO=<owner>/<repo>
MODEL_SUBDIR=<subdir>
./setup.sh weights           # downloads the configured target
./run-sglang-godspeed.sh start
```

Rules of thumb for a drop-in target:
- **Keep the architecture family.** The mamba/Gated-DeltaNet levers and the
  DSpark numbers are tuned for this exact hybrid arch. A different model
  (Llama, DeepSeek, …) breaks the assumptions: the drafter no longer applies
  and the mamba flags stop meaning anything.
- **Keep the drafter paired with the base it was distilled from.** A
  moderate fine-tune or re-quant of the *same* base keeps draft
  acceptance high. A heavily diverged fine-tune still runs, but the target's
  distribution drifts away from the drafts, so accept length and tok/s drop
  toward non-speculative. Recheck the `spec-accept-length` panel after any
  target swap.
- A fully different base model needs a matching drafter rebuild too (or run
  without DSpark and drop `--speculative-algorithm DSPARK`).

### GGUF is not a drop-in replacement

SGLang itself does have a GGUF path (`--load-format gguf` /
`--quantization gguf`, CUDA-supported in the `lmsysorg/sglang` image — a wide
weight-type list: Q4/Q5/Q8, K-quants, IQ series, unquant). But on **this**
stack it is *not* a drop-in replacement for the NVFP4 build:

- The speed win comes from the coordinated pipeline — custom Gated-DeltaNet /
  mamba kernels plus DSpark tuned to the NVFP4 target. A GGUF build dequantizes
  to plain weight tensors; there is no guarantee the Gated-DeltaNet kernels
  accept GGUF-dequantized weights, and that is the most likely break point.
- `--speculative-dspark-block-size` and the measured tok/s were tuned against
  the NVFP4 target's distribution — a GGUF target needs re-tuning, not just
  swapping.
- qwen3-arch GGUF loading has had known support gaps upstream
  (sgl-project/sglang #6281); verify against your image revision before
  spending a cold start on it.
- The only thing GGUF genuinely buys here is VRAM headroom (Q4/Q6 quants are
  smaller than W4A4 NVFP4 + FP8). Note that the model variant and the quant
  format are orthogonal: any other *safetensors* quant of the base (FP8,
  FP16, a re-quant) stays the clean drop-in documented above — no need to
  move to GGUF for that.

If you want VRAM headroom, the supported move is a lighter *safetensors*
quant of the same base (e.g. FP8 instead of NVFP4 W4A4), not a format change.

---

## Thinking / reasoning (Qwen3.8)

Qwen3.8-27B is a thinking model. The chat template exposes three knobs, all
passed to the server as SGLang's `--default-chat-template-kwargs` (a JSON
object applied to every request; a per-request `chat_template_kwargs` in the
API call still overrides the server default):

| knob | values | default | effect |
|---|---|---|---|
| `reasoning_effort` | `medium` (default) / `xhigh` / `low` | `medium` | how deeply the model reasons before answering. `xhigh` is the deepest; `low` is fastest/cheapest. (The template rejects any other string; "none"/"off" = set `enable_thinking=false`.) |
| `enable_thinking` | `true` / `false` | `true` | turn the thinking trace on/off entirely (off = no `reasoning_content`, 0 reasoning tokens). |
| `preserve_thinking` | `true` / `false` | `true` | keep the previous turn's thinking trace in the prompt (costs tokens, helps continued conversations). |

**Where it's set:** the launch scripts build `--default-chat-template-kwargs
"$DEFAULT_CHAT_TEMPLATE_KWARGS"`. That variable is `.env`-overridable; when
unset the launchers pass `{"reasoning_effort":"medium"}`, so **medium is the
default reasoning effort for this stack** (a good accuracy/speed balance for
day-to-day use). The template's own native default is `xhigh`; medium is
deliberately chosen here as the out-of-the-box trade-off.

> **Hint — pick your effort.** Leave it alone for `medium`. Want a deeper
> answer on hard tasks? Set `DEFAULT_CHAT_TEMPLATE_KWARGS='{"reasoning_effort":"xhigh"}'`
> in `.env`. Want it faster/cheaper? `"reasoning_effort":"low"` (or
> `"enable_thinking":false` to skip the trace entirely). Or override per-request
> with no restart — see below.

> **Measured at medium (the default).** With `reasoning_effort: medium` this
> stack peaks at **323 tok/s** on a live burst. The DSpark drafter is still
> what drives it — the live Prometheus source shows the drafter accepting ~3.3
> draft tokens/step at a ~0.33 accept rate (the same `spec_accept_*` series the
> dashboard's DSpark section graphs). So the out-of-the-box default is not just
> a good accuracy/speed balance — it's also the fastest thing you get without
> dropping to `low`, and it still has headroom to reach for `xhigh` when a
> task needs depth. (A rolling `max_over_time(gen_throughput[1h])` sits a bit
> lower than the burst peak because it averages idle between requests.)

Other `.env` examples:

```bash
# thinking off entirely (cheapest, no reasoning trace)
DEFAULT_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'
# fast + keep prior-turn thinking for ongoing chats
DEFAULT_CHAT_TEMPLATE_KWARGS='{"reasoning_effort":"low","preserve_thinking":true}'
```

Or override per-request (no restart) — a per-request `chat_template_kwargs`
wins over the server default:

```bash
curl -s http://localhost:8040/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-nvfp4","messages":[{"role":"user","content":"hi"}],
       "chat_template_kwargs":{"reasoning_effort":"xhigh"}}'
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

> **Later snapshot — medium default, 2026-08-20 ~21:00 CEST.** After the
> `medium` default went live (server boot ~20:45), the peak generation rate
> climbed to **323 tok/s** on a burst (the table above's `gen_throughput`
> max is the earlier, pre-restart idle snapshot). The drafter is still the
> driver: `sum(sglang:spec_accept_length)` ≈ 3.3 tokens/step, `spec_accept_rate`
> ≈ 0.33 — the same series the DSpark section graphs.

---

## Configuration (`.env`)

Everything is overridable in `.env` — no user-specific paths or usernames are
hardcoded. Common knobs: `SGGLANG_IMAGE`, `MODEL_REPO`/`DRAFTER_REPO`,
`MODEL_SUBDIR`/`DRAFTER_SUBDIR`, `HOST_PORT`, `API_KEY`, `MODELS_ROOT`,
`SERVED_MODEL_NAME` (id exposed at `/v1/models`; default
`qwen3.8-27b-nvfp4`), `DEFAULT_CHAT_TEMPLATE_KWARGS` (Qwen3.8 thinking knobs;
default `{"reasoning_effort":"medium"}`), `HF_HUB_ACCESS_TOKEN`,
`METRICS_HASH` (gateway auth). See `.env.example`.

---

## Troubleshooting / verify before you trust

- **Speed numbers**: ~260 tok/s is community-reported; the model card
  verifies ~180. The live **`medium` default measured a 323 tok/s burst peak**
  (see the Thinking section + the "Later snapshot" callout). Re-measure after
  boot (`status` + a timed decode) — a rolling 1h `gen_throughput` max sits
  below the burst because it averages idle time between requests.
- **OOM at boot** (vision preset): lower `--mem-fraction-static` to 0.80 and
  `--context-length` to ~120000.
- **DSpark vs MTP**: DSpark is the faster drop-in on this hybrid
  Gated-DeltaNet model (the in-checkpoint MTP head is the older K=1 path). No
  vision penalty with DSpark.
- **Metrics target down**: normal until a SGLang server is running; it flips up
  after `./run-sglang-*.sh start` reaches ready.
- **Prefill degrades after hours of mixed load**: community-observed — after
  ~4.5 h of multi-tenant use, short-ctx prefill slowed ~2.3× (T32 3.3 s →
  7.6 s) while short-ctx decode stayed fine; a plain container restart (same
  args, ~3 min cold start) fully restored baseline. If prefill TTFT creeps up
  on a long-running box, `./run-sglang-godspeed.sh start` (replaces the
  container) is the cheap first fix before suspecting the workload.

## Layout

```
sglang/
├── .env.example               # copy to .env (defaults work)
├── setup.sh                   # pull image + download target + drafter (+ optional bf16-drafter)
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

---

## Credit

Made with **AI — Qwen3.8** (the model this stack serves). Qwen3.8 drove the
SGLang tuning, chat-template work, Grafana dashboard, and monitoring
throughout; the Qwen3.8-27B checkpoint itself runs on this exact setup.
