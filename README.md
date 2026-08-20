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
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"hello"}]}'
```

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

---

## Configuration (`.env`)

Everything is overridable in `.env` — no user-specific paths or usernames are
hardcoded. Common knobs: `SGGLANG_IMAGE`, `MODEL_REPO`/`DRAFTER_REPO`,
`MODEL_SUBDIR`/`DRAFTER_SUBDIR`, `HOST_PORT`, `API_KEY`, `MODELS_ROOT`,
`HF_HUB_ACCESS_TOKEN`, `METRICS_HASH` (gateway auth). See `.env.example`.

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
├── dcgm-exporter/            # NVML -> Prometheus GPU exporter
└── models/                    # weights (created by setup.sh; gitignored)
```
