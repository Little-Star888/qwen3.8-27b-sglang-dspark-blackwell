# Changelog

User-facing changes to this stack, newest first. Dates are commit dates.

## 2026-08-30

- **DFlash2 is now the default recipe** (`run-sglang-dflash.sh` /
  `run-sglang-dflash-vision.sh`). The DSpark presets (`godspeed` / `vision`)
  are the alternative: higher burst ceiling, lower floor.
- `setup.sh` default now pulls the DFlash2 nightly image + target + DFlash2
  drafter; `./setup.sh dspark` is the new alternative path (DSpark image +
  drafter); `nvfp4-drafter` and `dflash2` subcommands kept.
- README rewritten around the DFlash2 default; the DFlash2-vs-DSpark
  measured comparison table (Grafana medians + peaks) moved up and now
  carries the default-vs-alternative framing.
- Text-only presets: `--mem-fraction-static` 0.90 → **0.88** (~0.6 GB OOM
  headroom; effective per-request ctx cap ~116K → ~98K tokens). Vision
  presets stay 0.82.
- DSpark vision preset: `--context-length` 150000 → **120000** (fits with
  headroom on the 32 GB 5090).
- Monitoring stack to current latest-stable: **Grafana 11.1.4 → 13.2.0**,
  **Prometheus v3.13.2 → v3.14.0** (Caddy 2.11.4 already current;
  dcgm-exporter remains the local build).
- **Docker hosts now work with no setup** (community PR): when `podman`
  is absent the scripts fall back to `bin/podman` / `bin/podman-compose`
  shims (translate `image exists` → `image inspect`, drop `--replace`,
  `--device nvidia.com/gpu=all` → `--gpus all`). Inert on Podman hosts —
  a real `podman` on `PATH` always wins. Self-test: `./bin/test-shim.sh`.
- Dashboard retitle: "DSpark, RTX5090" → "Qwen3.8-27B NVFP4 (RTX5090)";
  spec-decoding row now drafter-neutral (covers DFlash2 + DSpark).

## 2026-08-25

- **BF16 RadixArk DSpark drafter is the DSpark default**
  (`DRAFT_MODEL_QUANTIZATION=unquant`); the NVFP4 build became the opt-out
  (`./setup.sh nvfp4-drafter` + `modelopt_fp4` in `.env`).
- `--max-mamba-cache-size` default 5 → **8** (removes the long multi-turn
  eviction-recompute stall; ~5K token-pool cost). All sizing knobs are
  `.env`-overridable: `MAX_MAMBA_CACHE_SIZE`, `MEM_FRACTION_STATIC`,
  `DSPARK_BLOCK_SIZE`, `DRAFT_MODEL_QUANTIZATION`.
- README: target-model swap rules (safetensors quants are the clean
  drop-in; GGUF is not) and the generalized target-swap section.

## 2026-08-20

- `reasoning_effort: medium` becomes the stack default
  (`--default-chat-template-kwargs`); xhigh/low and per-request override
  documented.
- Thinking/reasoning knobs exposed (`enable_thinking`, `preserve_thinking`).
- Launchers are non-destructive: read-only VRAM check, never stop other
  containers or services.
- Stable served model id `qwen3.8-27b-nvfp4` via `--served-model-name`.
- Initial commit of the stack: prebuilt image, `setup.sh`, presets,
  monitoring (caddy/prometheus/grafana/dcgm), live-measured KPI proof.
