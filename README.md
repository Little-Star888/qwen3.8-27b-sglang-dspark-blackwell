# Qwen3.8-27B (NVFP4) on SGLang — DFlash2 speculative decoding

Run the 27B model on a single **RTX 5090 (Blackwell, 32 GB)** at fast
decode speed using SGLang + an NVFP4 checkpoint + the **DFlash2 drafter**
(z-lab block-diffusion) — the default recipe. The **DSpark drafter is the
alternative**: a bit more burst ceiling, a much lower floor. One
self-contained repo: prebuilt image, one `setup` command, four launch
presets (two default + two alternative), and an optional monitoring
dashboard.

```
./setup.sh                  # pull image + download weights (target + DFlash2 drafter)
./run-sglang-dflash.sh start
```

That's it. Open `http://localhost:8040/v1` (OpenAI-compatible) or
`/anthropic`.

[What changed, version by version: `CHANGELOG.md`](CHANGELOG.md).

---

## Key facts

| | |
|---|---|
| Model | Qwen3.8-27B, NVFP4 (weights) + lm_head in NVFP4 + FP8 KV cache |
| Draft | **DFlash2 drafter** (z-lab block-diffusion, separate ~3.6 GB BF16 model) — default · DSpark BF16 (~2.6 GB) — alternative |
| Engine | SGLang (`lmsysorg/sglang:nightly-dev-cu13-20260830-a1fe4e30`, prebuilt, no JIT) |
| GPU | 1× RTX 5090 (32 GB, sm_120 / Blackwell) |
| Target speed | default DFlash2: **median 221 tok/s, peak 277** (measured) · ~150–200 tok/s (vision) · DSpark alternative: ~300–323 tok/s burst ceiling, median 126 |
| Context | ~236k (text-only) / ~150k (vision default) / ~120k (DSpark vision) |
| Ports | API 8040 · gateway 8041 · Grafana 8042 · Prometheus 9091 (all LAN-reachable) |

The speed is the drafter: it proposes a block of tokens the big model
verifies in one step. The higher the accepted-draft rate, the faster the
decode. DFlash2 drafts a whole block diffusively in one denoise step,
which is what lifts the *median*; this repo exposes all of it live in the
Grafana dashboard.

---

## Feature overview

| | |
|---|---|
| ⚡ Inference | Prebuilt SGLang image — boots with **no JIT/compile**; NVFP4 weights + FP8 KV cache, Blackwell SM120 FlashInfer FP4 GEMM |
| 🎲 Speculative decoding | **Default: DFlash2 drafter** proposes a whole block of draft tokens per denoise step; the 27B model **verifies the block in one step** (block 8, draft window 16K). **Alternative: DSpark drafter** proposes autoregressively (block 7 text / 5 vision) |
| 👁️ Four presets | **dflash** (default, text-only, 236k ctx) ⇄ **dflash-vision** (default, vision ON, 150k ctx) ⇄ **godspeed** (DSpark alt, text-only) ⇄ **vision** (DSpark alt) — swappable on one GPU |
| 📊 Grafana | Auto-provisioned dashboard: dense KPI tiles + a live **speculative-decoding win-rate** section (graphs the `spec_accept_*` series either drafter exports). `make_dashboard.py` is the source; the JSON is build output |
| 📈 Prometheus | 30-day retention, scrapes SGLang + DCGM — the dashboard's own data source (`:9091`, LAN-reachable; a "Prometheus targets" row shows target health) |
| 🎛️ GPU telemetry | DCGM exporter (util / mem / temp / power) alongside `sglang:` engine metrics; every tile aggregated (`sum()`/`max()`) to a single value |
| 🔐 Gateway | Caddy **Basic-auth** on 8041 → OpenAI/Anthropic endpoints with bearer `API_KEY`; auth-off by default |
| 🐳 GPU isolation | Rootless **Podman + NVIDIA CDI** (`--device nvidia.com/gpu=all`); boot does a read-only VRAM check — it never stops other services |
| 🏷️ Stable API id | `qwen3.8-27b-nvfp4` via `--served-model-name` (overridable in `.env`), so `/v1/models` never leaks the `/model` path |
| 🧠 Thinking control | Qwen3.8 thinking/reasoning knobs via SGLang `--default-chat-template-kwargs`: `reasoning_effort` (medium default / xhigh / low), `enable_thinking`, `preserve_thinking` — set in `.env` as `DEFAULT_CHAT_TEMPLATE_KWARGS` |
| 📦 Self-contained | One `./setup.sh` (image + target + DFlash2 drafter → `./models`); the DSpark alternative adds `./setup.sh dspark` (own image + drafter); share across stacks via `MODELS_ROOT` |

---

## Requirements

- **Linux x86_64**, one RTX 5090, NVIDIA driver ≥ 580 (Blackwell).
- **Podman** (rootless) + the NVIDIA **CDI** device plugin — so `podman run
  --device nvidia.com/gpu=all` works — plus **podman-compose** for the
  monitoring stack. **Or Docker**, no setup needed: with podman absent the
  scripts fall back to `bin/podman` / `bin/podman-compose`, which translate the
  three calls that aren't 1:1 (`image exists` → `image inspect`, drop
  `--replace`, `--device nvidia.com/gpu=all` → `--gpus all`). Docker needs
  nvidia-container-toolkit; a real podman on `PATH` still takes precedence.
- **Python `hf` CLI** (huggingface_hub) for weight downloads.
- ~34 GB for the default image + ~22 GB for weights (target 18 GB + DFlash2
  drafter ~3.6 GB). The DSpark alternative adds its own 41 GB image +
  ~2.6 GB drafter only if you also run `./setup.sh dspark`.

---

## 1 · Set up

```bash
git clone <this-repo> sglang && cd sglang
cp .env.example .env            # defaults are fine to start
./setup.sh                      # pulls the image, downloads target + DFlash2 drafter
```

Want the **DSpark alternative** presets (`godspeed` / `vision`) too? They use
a different SGLang image and a different drafter:

```bash
./setup.sh dspark               # pulls the DSpark image + downloads the DSpark BF16 drafter
```

Weights go to `./models/` by default (self-contained). To share one store
across multiple stacks, set `MODELS_ROOT` in `.env` and `setup.sh` symlinks
into it.

## 2 · Launch

```bash
./run-sglang-dflash.sh start         # default: DFlash2, text-only, max ctx, steadiest speed
# or
./run-sglang-dflash-vision.sh start  # default: DFlash2, vision ON, lower ctx

./run-sglang-dflash.sh status        # wait until "HTTP 200 (ready)"
```

DSpark alternative:

```bash
./run-sglang-godspeed.sh start       # DSpark, text-only
# or
./run-sglang-vision.sh start         # DSpark, vision ON
```

Start does a read-only VRAM check and never stops other containers or
services — if the 5090 is busy (e.g. a vLLM stack), free it manually first.
All four presets share one container name, so swap by `stop`-ing one,
`start`-ing the other — you only have one GPU, so they're **swappable, not
simultaneous**.

## 3 · Call it

```bash
curl -s http://localhost:8040/v1/models
curl -s http://localhost:8040/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b-nvfp4","messages":[{"role":"user","content":"hello"}]}'
```

The served id is `qwen3.8-27b-nvfp4` (all presets pass
`--served-model-name`, so `/v1/models` no longer leaks the container path
`/model`). Override it per-stack via `SERVED_MODEL_NAME` in `.env`.

---

## Presets (default: DFlash2 · alternative: DSpark)

| | **dflash** (default, text-only) | **dflash-vision** (default) | **godspeed** (DSpark alt, text-only) | **vision** (DSpark alt) |
|---|---|---|---|---|
| Drafter | DFlash2 z-lab (~3.6 GB BF16) | DFlash2 | DSpark BF16 RadixArk (~2.6 GB) | DSpark BF16 |
| Image | nightly (DFlash2) | nightly | `qwen38-27b` | `qwen38-27b` |
| `--context-length` | 237568 | 150000 | 237568 | 120000 |
| `--mem-fraction-static` | 0.91 | 0.82 | 0.88 | 0.82 |
| `--max-mamba-cache-size` | 8 | 8 | 8 | 8 |
| Draft flags | block 8 + draft window 16384 | block 8 + draft window 16384 | block 7 | block 5 |
| Expected decode | median **221** · peak **277 tok/s** | ~150–200 tok/s | median **126** · peak **~300 tok/s** (323 burst, earlier window) | ~150–200 tok/s |
| VRAM | ~30.6 GB | ~30 GB | ~30.6 GB | ~30 GB |

Shared flags: `--kv-cache-dtype fp8_e4m3`, `--attention-backend flashinfer`
(SM120), mamba linear-attention levers (`--mamba-ssm-dtype bfloat16`,
`--max-mamba-cache-size`), `--max-running-requests 1`. The DFlash2 presets
pass `--mamba-radix-cache-strategy extra_buffer` (the image hard-asserts
that `extra_buffer_lazy` is unsupported with DFLASH); the DSpark presets use
`extra_buffer_lazy`. Sizing and drafter knobs are `.env`-overridable:
`MAX_MAMBA_CACHE_SIZE` (default 8), `MEM_FRACTION_STATIC`,
`DFLASH_BLOCK_SIZE`, `DRAFT_WINDOW_SIZE` (DFlash2); `DSPARK_BLOCK_SIZE`,
`DRAFT_MODEL_QUANTIZATION` (DSpark) — see `.env.example`.

**DFlash2 is the default** because it's the *steady* winner: its median
throughput is ~1.8× the DSpark median and its draft acceptance is far more
consistent (see the comparison below). DSpark remains the alternative when
you want its higher burst ceiling.

### DFlash2 vs DSpark (measured, same main checkpoint)

`Qwen3.8-27B-NVFP4-RTX5090-LMHead4` target in both cases; numbers =
median / peak of the Grafana data source, DSpark godspeed Aug 25–30 window,
DFlash2 since Aug 30 ~09:00 UTC:

| | DSpark godspeed (BF16, block 7) | DFlash2 (block 8, draft window 16384) |
|---|---|---|
| Throughput (tok/s) | median **126** · peak **299** | median **221** · peak **277** |
| Draft accept length (tok/step) | median **2.6** · peak **5.25** | median **4.0** · peak **5.4** |
| Accept rate (accepted/proposed) | median **0.23** · peak **0.61** | median **0.43** · peak **0.62** |
| TTFT p50 | 0.51 s | 0.47 s |
| Inter-token latency p50 | ~10 ms | ~0–4 ms |
| VRAM (mem-fraction) | 0.88 → ~30.6 GB | 0.88 → ~30.6 GB |

DFlash2 trades DSpark's ~300 tok/s burst ceiling for a **much higher
floor**: the median throughput is ~1.8× the DSpark median and the
accept-length median goes 2.6 → 4.0 — the win is steadiness, not peaks. The
two windows are not equal in size (the DFlash2 sample is the shorter one, a
few hours of live traffic vs days), so read the DFlash2 medians as "at or
above DSpark" rather than a head-to-head benchmark.

### Alternative: DSpark drafter (godspeed / vision presets)

The DSpark family proposes blocks **autoregressively** (token after token),
vs DFlash2's one-step denoise. Default dtype is the ~2.6 GB **BF16**
RadixArk build (`DRAFT_MODEL_QUANTIZATION=unquant`) — community testing on
the 5090 (single-request, thinking ON) showed it drafts noticeably better
than the ~1.4 GB **NVFP4** build: accept length ~4.0 (~171 tok/s) vs ~1.7
(~93 tok/s), so BF16 is the default. `./setup.sh dspark` downloads it
alongside the DSpark image.

To opt out to the smaller NVFP4 DSpark drafter (saves ~1.1 GB of VRAM):

```bash
./setup.sh nvfp4-drafter       # downloads gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4
# in .env:
DRAFTER_SUBDIR=Qwen3.8-27B-DSpark-NVFP4
DRAFT_MODEL_QUANTIZATION=modelopt_fp4
```

Restart the preset afterwards (`./run-sglang-godspeed.sh start`). The
DSpark image's `--speculative-draft-model-quantization` accepts `unquant`
(BF16, the default), `modelopt_fp4` (NVFP4), and the usual SGLang quant
list.

> DFlash2 needs the newer nightly image (`DFLASH_SGLANG_IMAGE`):
> DFlash2DraftModel only landed upstream 2026-08-19, and the DSpark image
> (`qwen38-27b`, 2026-08-14) predates it. The two recipe images are
> independent; each preset pulls its own.

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

### Context pool: `--context-length` vs the real token pool

Two different numbers are both called "context length" and they are not the
same thing:

- **`--context-length`** (nominal). Set to `237568` (text-only). This is the
  *architectural* cap — the longest sequence the model can address. It costs
  almost no VRAM; it's a ceiling, not a budget. A 230K request still has to
  physically fit in the KV pool below or it OOMs.
- **The KV token pool** (`sglang:max_total_num_tokens`). This is the *physical*
  limit: how many KV cache slots actually exist in VRAM. It is recomputed from
  free VRAM at every container start, and it is the number your real requests
  must fit in (prompt + generated tokens ≤ pool).

On this 32 GB 5090 the pool is driven by `--mem-fraction-static` (how much
VRAM SGLang may hold for weights + KV + drafters) and by the drafter's own
draft KV pool. The measured relationship is stable across boots:

**pool ≈ `kv_cache_memory_GB` × 32,768 tokens/GB** (FP8_e4m3 cache, ≈32 B/token;
measured 32,763–32,772 across every boot this month — the ratio is stable, so
the table below is a prediction you can check against `/metrics`).

Measured on this box (peak → current → target):

| Stack | mfs | drafter | KV GB | **token pool** |
|---|---|---|---|---|
| peak | 0.90 | NVFP4 DSpark | 5.15 | **~169K** |
| legacy (≤ 08-29) | 0.88 | DFlash2 (draft window 16384) | 3.00 | **~98K** |
| **default (128K)** | 0.91 | DFlash2 (draft window 16384) | 3.73 | **~122K** (measured 122,069 on 2026-08-30; boot headroom 0.85 GB) |

**Levers, in order of preference** (each frees VRAM → grows the pool;
verify the real number with `sglang:max_total_num_tokens` after each boot):

| Lever | `.env` knob | Effect |
|---|---|---|
| 1. Draft window | `DRAFT_WINDOW_SIZE` | DFlash2's own draft KV pool scales with the window. 16384 → 8192 is the least-disruptive first cut (keeps 16K of draft context — plenty for agent sessions); halving it frees roughly half the draft pool's share of the budget. |
| 2. Mamba slots | `MAX_MAMBA_CACHE_SIZE` | 8 → 5 frees ~5K of the pool; drop only for single-shot long-ctx work. |
| 3. Mem fraction | `MEM_FRACTION_STATIC` | Primary pool dial. Higher = bigger pool, less OOM headroom at boot. |

### The 128K default (and how to move it)

The script default since 2026-08-30 is the **128K recipe**: `MEM_FRACTION_STATIC=0.91`
+ DFlash2 draft window 16384, measured pool **~122K** (122,069 on the
2026-08-30 boot; boot headroom 0.85 GB — slightly under the ~1 GB comfort
line, but clean). It fixes the failure
mode the older 0.88 default had — a ~98K pool OOMs a ~75K-context agent session
mid-decode once `max_tokens` pushes it past the pool.

```bash
# .env overrides (script default is the 128K recipe; only set these to move away)
MEM_FRACTION_STATIC=0.91       # 128K default; 0.88 → ~98K, 0.89 → OOM-backoff
DRAFT_WINDOW_SIZE=16384        # extra lever: 8192 shrinks the draft pool (more headroom)
```

**Verify the real number after every boot** — read
`sglang:max_total_num_tokens` from `/metrics` (Grafana panel "KV token pool").
If it lands under 128K, lower `DRAFT_WINDOW_SIZE` (16384 → 8192) or raise mfs;
if it OOMs at boot (headroom < ~1 GB), back off mfs to 0.89.

**Companion (client side):** whatever pool you land on, the client's
`context_length` cap for this endpoint must track it — otherwise the client
believes the window is 237K and only compresses near ~118K, which is *past*
where a mid-decode OOM actually happens. In Hermes, set
`custom_providers[].models.qwen3.8-27b-nvfp4.context_length` to the measured
pool (`128000` for the 128K default, `98000` for the legacy 0.88 setup) in
`~/.hermes/config.yaml`.

### Swapping the target model (safetensors)

The target model and the drafter are two **independent** mounts, wired by
independent `.env` knob pairs:

| Role | `.env` keys | mount | server flag |
|---|---|---|---|
| Target (main model) | `MODEL_REPO` / `MODEL_SUBDIR` | `/model` | `--model-path /model` |
| DFlash2 drafter | `DFLASH_DRAFTER_SUBDIR` | `/model_dflash` | `--speculative-draft-model-path /model_dflash` |
| DSpark drafter | `DRAFTER_REPO` / `DRAFTER_SUBDIR` | `/model_dspark` | `--speculative-draft-model-path /model_dspark` |

So the target is not tied to the drafter (or vice versa):

```bash
# e.g. another quant of the same base (FP8, FP16, a re-quant or re-merge),
# or any other Qwen3.8-27B safetensors build. `setup.sh` downloads it and
# symlinks it into ./models/.
MODEL_REPO=<owner>/<repo>
MODEL_SUBDIR=<subdir>
./setup.sh weights           # downloads the configured target
./run-sglang-dflash.sh start
```

Rules of thumb for a drop-in target:
- **Keep the architecture family.** The mamba/Gated-DeltaNet levers and the
  drafter numbers are tuned for this exact hybrid arch. A different model
  (Llama, DeepSeek, …) breaks the assumptions: the drafter no longer
  applies and the mamba flags stop meaning anything.
- **Keep the drafter paired with the base it was distilled from.** A
  moderate fine-tune or re-quant of the *same* base keeps draft
  acceptance high. A heavily diverged fine-tune still runs, but the target's
  distribution drifts away from the drafts, so accept length and tok/s drop
  toward non-speculative. Recheck the `spec-accept-length` panel after any
  target swap.
- A fully different base model needs a matching drafter rebuild too (or run
  without speculation and drop the `--speculative-algorithm` flag).

### GGUF is not a drop-in replacement

SGLang itself does have a GGUF path (`--load-format gguf` /
`--quantization gguf`, CUDA-supported in the `lmsysorg/sglang` image — a
wide weight-type list: Q4/Q5/Q8, K-quants, IQ series, unquant). But on
**this** stack it is *not* a drop-in replacement for the NVFP4 build:

- The speed win comes from the coordinated pipeline — custom Gated-DeltaNet /
  mamba kernels plus the drafters tuned to the NVFP4 target. A GGUF build
  dequantizes to plain weight tensors; there is no guarantee the Gated-DeltaNet
  kernels accept GGUF-dequantized weights, and that is the most likely break
  point.
- `--speculative-dflash-block-size` / `--speculative-dspark-block-size` and
  the measured tok/s were tuned against the NVFP4 target's distribution — a
  GGUF target needs re-tuning, not just swapping.
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

> **Measured at medium (the default).** With `reasoning_effort: medium` the
> default DFlash2 recipe peaks at **277 tok/s** on a live burst and holds a
> **221 tok/s median** across its window — the drafter accepts ~4.0 draft
> tokens/step at a ~0.43 accept rate (the same `spec_accept_*` series the
> dashboard's speculative-decoding section graphs). The DSpark alternative
> peaked a touch higher (323 tok/s burst in the earlier window) but ran
> ~126 tok/s median, so the out-of-the-box default is not just a good
> accuracy/speed balance — it's the steadiest fast thing you get without
> dropping to `low`, with headroom to reach for `xhigh` when a task needs
> depth. (A rolling `max_over_time(gen_throughput[1h])` sits a bit lower
> than the burst peak because it averages idle between requests.)

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
  -H "Authorization: Bearer ***" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-nvfp4","messages":[{"role":"user","content":"hi"}],
       "chat_template_kwargs":{"reasoning_effort":"xhigh"}}'
```

---

## Monitoring (optional)

Prometheus + Grafana + Caddy + a GPU exporter, all isolated (own ports/network/
volumes). A dedicated **speculative-decoding section** shows live
accepted-draft rate — it graphs the `sglang:spec_*` series, so it tracks
*whichever* drafter is running (the section's generated title still says
"DSpark"; it's a label, the series are drafter-agnostic).

```bash
./monitor.sh up           # start the monitoring stack
./monitor.sh status        # target health (sglang target is up only while a server runs)
./monitor.sh dashboard     # print the dashboard URL + login hints
```

Dashboard: `http://localhost:8042` → folder *sglang* → "SGLang — Qwen3.8-27B
(RTX5090)" (the generated title currently reads "DSpark, RTX5090"). Watch
`spec_accept_length` / `spec_accept_rate` to see the drafter win-rate that
drives your throughput.

### Live proof (measured via the dashboard's own data source)

Every value below was read through the same Prometheus (port 9091) the
Grafana panels query, on 2026-08-20 20:11 CEST with `sglang-qwen38` up
33 min on the DSpark godspeed preset (the default back then), idle after a
few smoke-test requests:

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

> **Later snapshot — medium default, 2026-08-20 ~21:00 CEST (DSpark era).**
> After the `medium` default went live (server boot ~20:45), the peak
> generation rate climbed to **323 tok/s** on a burst (the table above's
> `gen_throughput` max is the earlier, pre-restart idle snapshot). The
> drafter is still the driver: `sum(sglang:spec_accept_length)` ≈ 3.3
> tokens/step, `spec_accept_rate` ≈ 0.33 — the same series the
> speculative-decoding section graphs. (Under the current DFlash2 default,
> the corresponding live numbers are the 221 median / 277 peak / ~4.0 /
> ~0.43 above.)

---

## Configuration (`.env`)

Everything is overridable in `.env` — no user-specific paths or usernames
are hardcoded. Common knobs: `DFLASH_SGLANG_IMAGE` (default-recipe image),
`MODEL_REPO`/`DFLASH_DRAFTER_SUBDIR` (default recipe), `SGGLANG_IMAGE` /
`DRAFTER_REPO`/`DRAFTER_SUBDIR` / `DRAFT_MODEL_QUANTIZATION` /
`DSPARK_BLOCK_SIZE` (DSpark alternative), `DFLASH_BLOCK_SIZE` /
`DRAFT_WINDOW_SIZE`, `HOST_PORT`, `API_KEY`, `MODELS_ROOT`,
`SERVED_MODEL_NAME` (id exposed at `/v1/models`; default
`qwen3.8-27b-nvfp4`), `DEFAULT_CHAT_TEMPLATE_KWARGS` (Qwen3.8 thinking knobs;
default `{"reasoning_effort":"medium"}`), `HF_HUB_ACCESS_TOKEN`,
`METRICS_HASH` (gateway auth). See `.env.example`.

---

## Troubleshooting / verify before you trust

- **Speed numbers**: the default DFlash2 recipe measured **221 tok/s median /
  277 peak** (see the comparison table above); the DSpark alternative peaks
  ~300–323 but runs ~126 median. Re-measure after boot (`status` + a timed
  decode) — a rolling 1h `gen_throughput` max sits below the burst because
  it averages idle time between requests.
- **OOM at boot** (vision presets): lower `--mem-fraction-static` to 0.80
  and `--context-length` to ~120000 (dflash-vision); on DFlash2 text-only
  you can also lower `DRAFT_WINDOW_SIZE` — it's the one knob that bounds the
  DFlash2 draft KV pool.
- **DFlash2 vs DSpark vs MTP**: DFlash2 is the default (steady, high floor);
  DSpark is the alternative (higher burst ceiling); the in-checkpoint MTP
  head is the older K=1 path and is not competitive on this hybrid
  Gated-DeltaNet model. No vision penalty with either drafter.
- **Metrics target down**: normal until a SGLang server is running; it flips
  up after `./run-sglang-*.sh start` reaches ready.
- **Prefill degrades after hours of mixed load**: community-observed — after
  ~4.5 h of multi-tenant use, short-ctx prefill slowed ~2.3× (T32 3.3 s →
  7.6 s) while short-ctx decode stayed fine; a plain container restart (same
  args, ~3 min cold start) fully restored baseline. If prefill TTFT creeps up
  on a long-running box, `./run-sglang-dflash.sh start` (replaces the
  container) is the cheap first fix before suspecting the workload.

## Layout

```
sglang/
├── .env.example               # copy to .env (defaults work)
├── CHANGELOG.md               # user-facing changes, newest first
├── setup.sh                   # pull image + download target + DFlash2 drafter (+ dspark alternative)
├── run-sglang-dflash.sh       # DEFAULT preset, text-only (start|stop|logs|status)
├── run-sglang-dflash-vision.sh# DEFAULT preset, vision ON
├── run-sglang-godspeed.sh     # DSpark alternative, text-only
├── run-sglang-vision.sh       # DSpark alternative, vision ON
├── monitor.sh                 # monitoring up|down|status|dashboard
├── bin/                       # podman -> docker shims (auto-used when podman is absent)
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
