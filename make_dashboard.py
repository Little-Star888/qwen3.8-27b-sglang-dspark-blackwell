#!/usr/bin/env python3
"""Generate the SGLang monitoring dashboard JSON for Grafana.

Mirrors the vLLM dashboard structure (stat KPIs -> timeseries -> GPU -> errors),
reworked for SGLang's real `sglang:`-prefixed metric names, and adds a
DSpark speculative-decoding section (the drafter win-rate you care about).

Gauge / counter / histogram types confirmed against
python/sglang/srt/observability/metrics_collector.py.

Output: grafana/dashboards/sglang-qwen38-27b.json  (schemaVersion 39)
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))

P = "prometheus"
COL = "rgb(31, 115, 182)"

def stat(title, expr, unit=None, decimals=0, span=4, threshold=None):
    p = {
        "type": "stat", "title": title, "gridPos": {"h": 3, "w": span, "x": 0, "y": 0},
        "datasource": {"type": "prometheus", "uid": P},
        "fieldConfig": {"defaults": {"unit": unit or "none", "decimals": decimals,
                                     "min": 0,
                                     "thresholds": {"mode": "absolute",
                                                    "steps": (threshold or [{"color": "green", "value": None}])}},
                        "overrides": []},
        "options": {"colorMode": "value", "graphMode": "none", "orientation": "auto",
                    "reduceOptions": {"calcs": ["lastNotNull"]}, "textMode": "value",
                    "justifiedValue": True},
        "targets": [{"datasource": {"type": "prometheus", "uid": P}, "expr": expr,
                     "legendFormat": title, "refId": "A", "editorMode": "code"}],
    }
    return p

def timeseries(title, exprs, unit="none", span=24, height=6, min_span=0):
    # exprs: list of (expr, legend)
    t = [
        {"datasource": {"type": "prometheus", "uid": P}, "expr": e, "legendFormat": l,
         "refId": chr(65 + i), "editorMode": "code"} for i, (e, l) in enumerate(exprs)
    ]
    return {
        "type": "timeseries", "title": title, "gridPos": {"h": height, "w": span, "x": 0, "y": 0},
        "datasource": {"type": "prometheus", "uid": P},
        "fieldConfig": {"defaults": {"unit": unit, "min": min_span,
                                     "custom": {"drawStyle": "line", "lineWidth": 1,
                                                "fillOpacity": 8, "spanNulls": False,
                                                "showPoints": "never", "stacking": {"mode": "normal"}}},
                        "overrides": []},
        "options": {"legend": {"calcs": ["mean", "max"], "displayMode": "list", "placement": "bottom",
                               "showLegend": True},
                    "tooltip": {"mode": "multi", "sort": "desc"}},
        "targets": t,
    }

def row(title):
    return {"type": "row", "title": title, "collapsed": False,
            "gridPos": {"h": 1, "w": 24, "x": 0, "y": 0}, "panels": []}

def row_collapsed(title, panels):
    return {"type": "row", "title": title, "collapsed": True,
            "gridPos": {"h": 1, "w": 24, "x": 0, "y": 0}, "panels": panels}

def gauge(title, expr, maxv=1.0):
    p = stat(title, expr, unit=None, span=6,
             threshold=[{"color": "green", "value": None},
                       {"color": "red", "value": 0.9}])
    p["type"] = "gauge"
    p["fieldConfig"]["defaults"]["max"] = maxv
    p["options"] = {"colorMode": "value", "graphMode": "area", "orientation": "auto",
                    "showThresholdLabels": False, "showThresholdMarkers": True}
    return p

def q(metric, agg="avg", window=""):
    """avg_over_time / rate helper is inline; this is a convenience placeholder."""
    return metric

panels = []

# ---- KPI stats (row) ----
# All 6 KPI tiles restored (they are the at-a-glance snapshot). Each expr is
# aggregated to a single series because this SGLang build exports several
# metrics with multiple label sets (is_streaming / mode / phase), which
# would otherwise stack one value per series in the tile.
r = row("Live KPIs"); 
k = [
    ("Generation rate (tok/s)", 'sum(sglang:gen_throughput)', None, 0, 4,
     [{"color":"green","value":None},{"color":"orange","value":0},{"color":"red","value":0.0001}]),
    ("Requests running", 'sum(sglang:num_running_reqs)', None, 0, 4,
     [{"color":"green","value":None}]),
    ("Requests waiting (queue)", 'sum(sglang:num_queue_reqs)', None, 0, 4,
     [{"color":"green","value":None},{"color":"red","value":1}]),
    ("KV cache used tokens", 'sum(sglang:kv_used_tokens)', None, 0, 4,
     [{"color":"green","value":None}]),
    ("Mamba/linear-attn usage", 'max(sglang:mamba_usage)', "percentunit", 2, 4,
     [{"color":"green","value":None},{"color":"orange","value":0.8},{"color":"red","value":0.95}]),
    ("GPU util %", 'max(DCGM_FI_DEV_GPU_UTIL)', "percent", 0, 4,
     [{"color":"green","value":None},{"color":"orange","value":90}]),
]
for t,e,u,d,span,th in k:
    r["panels"].append(stat(t,e,unit=u,decimals=d,span=span,threshold=th))
panels.append(r)

# ---- Speculative decoding (drafter) ----
rs = row("Speculative decoding (drafter)")
rs["panels"] += [
    stat("Drafts accepted / step", 'sum(sglang:spec_accept_length)', None, 3, 4,
         [{"color":"orange","value":None},{"color":"green","value":2},{"color":"green","value":4}]),
    stat("Accept rate (acc/proposed)", 'sum(sglang:spec_accept_rate)', "percentunit", 3, 4,
         [{"color":"red","value":None},{"color":"orange","value":0.5},{"color":"green","value":0.8}]),
    stat("Block accept length (uncapped)", 'sum(sglang:spec_block_accept_length)', None, 3, 4,
         [{"color":"green","value":None}]),
    stat("Active block size", 'sum(sglang:spec_num_steps)', None, 0, 4,
         [{"color":"blue","value":None}]),
    stat("Spec verify calls (total)", 'sum(sglang:spec_verify_calls_total)', "none", 0, 4,
         [{"color":"blue","value":None}]),
    stat("Draft tokens (total)", 'sum(sglang:spec_num_draft_tokens)', "none", 0, 4,
         [{"color":"blue","value":None}]),
]
rs["panels"].append(
    timeseries("Speculative decoding over time", [
        ("sum(sglang:spec_accept_length)", "accept length (tok/step)"),
        ("sum(sglang:spec_accept_rate) * 100", "accept rate %"),
        ("sum(sglang:spec_block_accept_length)", "block accept length"),
        ("sum(sglang:spec_num_draft_tokens)", "draft tokens"),
    ], unit="none")
)
panels.append(rs)

# ---- Latency ----
rl = row("Latency")
rl["panels"] += [
    timeseries("Time to first token (p50 / p95 / p99)", [
        ("histogram_quantile(0.5, sum(rate(sglang:time_to_first_token_seconds_bucket[5m])) by (le))", "TTFT p50"),
        ("histogram_quantile(0.95, sum(rate(sglang:time_to_first_token_seconds_bucket[5m])) by (le))", "TTFT p95"),
        ("histogram_quantile(0.99, sum(rate(sglang:time_to_first_token_seconds_bucket[5m])) by (le))", "TTFT p99"),
    ], unit="s", span=8, height=6),
    timeseries("Inter-token latency (p50 / p95)", [
        ("histogram_quantile(0.5, sum(rate(sglang:inter_token_latency_seconds_bucket[5m])) by (le))", "TPOT p50"),
        ("histogram_quantile(0.95, sum(rate(sglang:inter_token_latency_seconds_bucket[5m])) by (le))", "TPOT p95"),
    ], unit="s", span=8, height=6),
    timeseries("E2E request latency (p50 / p95 / p99)", [
        ("histogram_quantile(0.5, sum(rate(sglang:e2e_request_latency_seconds_bucket[5m])) by (le))", "E2E p50"),
        ("histogram_quantile(0.95, sum(rate(sglang:e2e_request_latency_seconds_bucket[5m])) by (le))", "E2E p95"),
        ("histogram_quantile(0.99, sum(rate(sglang:e2e_request_latency_seconds_bucket[5m])) by (le))", "E2E p99"),
    ], unit="s", span=8, height=6),
]
panels.append(rl)

# ---- Throughput ----
rt = row("Throughput")
rt["panels"] += [
    timeseries("Token throughput (tok/s)", [
        ("sum(sglang:gen_throughput)", "gen tok/s"),
    ], unit="tok/s", span=12, height=6),
    timeseries("Forward passes / queue (scheduler)", [
        ("sum(rate(sglang:cuda_graph_passes_total[5m]))", "CUDA-graph fwd passes /s"),
        ("sum(sglang:num_queue_reqs)", "queued reqs"),
    ], unit="none", span=12, height=6),
    timeseries("Tokens (input / generated, since start)", [
        ("sum(sglang:prompt_tokens_total)", "input tokens"),
        ("sum(sglang:generation_tokens_total)", "generated tokens"),
    ], unit="none", span=12, height=6),
]
panels.append(rt)

# ---- Cache / KV ----
rc = row("KV cache & prefix cache")
rc["panels"] += [
    stat("KV token usage", 'sum((sglang:kv_used_tokens + sglang:kv_evictable_tokens) / sglang:max_total_num_tokens)', "percentunit", 2, 8,
         [{"color":"green","value":None},{"color":"orange","value":0.8},{"color":"red","value":0.95}]),
    stat("Prefix cache hit rate", 'sum(sglang:cache_hit_rate)', "percentunit", 2, 8,
         [{"color":"red","value":None},{"color":"orange","value":0.5},{"color":"green","value":0.8}]),
    stat("KV evictable tokens", 'sum(sglang:kv_evictable_tokens)', "none", 0, 8,
         [{"color":"green","value":None}]),
    timeseries("KV used / available tokens", [
        ("sum(sglang:kv_used_tokens)", "used"),
        ("sum(sglang:kv_available_tokens)", "available"),
        ("sum(sglang:kv_evictable_tokens)", "evictable"),
    ], unit="none", span=12, height=6),
    timeseries("Cached tokens & evictions (rate)", [
        ("sum(rate(sglang:cached_tokens_total[5m]))", "cached tok/s"),
        ("sum(rate(sglang:evicted_tokens_total[5m]))", "evicted tok/s"),
    ], unit="none", span=12, height=6),
]
panels.append(rc)

# ---- Requests / errors ----
rr = row("Requests & errors")
rr["panels"] += [
    stat("Requests (total)", 'sum(sglang:num_requests_total)', "none", 0, 6,
         [{"color":"blue","value":None}]),
    stat("Aborted (1h)", 'increase(sglang:num_aborted_requests_total[1h]) or vector(0)', "none", 0, 6,
         [{"color":"green","value":None},{"color":"red","value":1}]),
    stat("Retracted (1h)", 'sum(increase(sglang:num_retracted_reqs[1h])) or vector(0)', "none", 0, 6,
         [{"color":"green","value":None},{"color":"red","value":1}]),
    stat("Queue wait p95 (5m)", 'histogram_quantile(0.95, sum(rate(sglang:queue_time_seconds_bucket[5m])) by (le))', "s", 3, 6,
         [{"color":"green","value":None}]),
    timeseries("Request rate (finished / aborted)", [
        ("sum(rate(sglang:num_requests_total[1m]))", "finished rps"),
        ("sum(rate(sglang:num_aborted_requests_total[1m])) or vector(0)", "aborted rps"),
    ], unit="ops", span=24, height=6),
]
panels.append(rr)

# ---- GPU (DCGM) ----
rg = row("GPU (DCGM)")
rg["panels"] += [
    timeseries("GPU utilization (DCGM)", [
        ("DCGM_FI_DEV_GPU_UTIL", "util %"),
    ], unit="percent", span=6, height=5),
    timeseries("GPU memory (MiB)", [
        ("DCGM_FI_DEV_MEMORY_USED", "used MiB"),
        ("DCGM_FI_DEV_MEMORY_TOTAL", "total MiB"),
    ], unit="Mbyte", span=6, height=5),
    timeseries("GPU power (W)", [
        ("DCGM_FI_DEV_POWER_USAGE", "power W"),
    ], unit="watt", span=6, height=5),
    timeseries("GPU temperature (C)", [
        ("DCGM_FI_DEV_GPU_TEMP", "temp C"),
    ], unit="celsius", span=6, height=5),
]
panels.append(rg)

# ---- Startup / misc (collapsed) ----
# Both startup metrics are exported once per phase label (load_weight,
# kv_cache_allocation, scheduler_e2e, ...), so sum them into one total.
rm = row_collapsed("Startup & misc", [
    stat("CUDA graph build (s)", 'sum(sglang:startup_cuda_graph_time_seconds)', "s", 1, 4,
         [{"color":"green","value":None}]),
    stat("Weight VRAM (GB)", 'max(sglang:weight_memory_usage_gb)', "GiB", 1, 4,
         [{"color":"blue","value":None}]),
    stat("KV cache VRAM (GB)", 'max(sglang:kv_cache_memory_usage_gb)', "GiB", 1, 4,
         [{"color":"blue","value":None}]),
    stat("Startup time (s)", 'sum(sglang:startup_time_seconds)', "s", 1, 4,
         [{"color":"blue","value":None}]),
])
panels.append(rm)

# The builders nest each row's panels inside the row object ("panels" key).
# Grafana wants an OPEN row's panels as TOP-LEVEL siblings (nested on an open
# row = empty/collapsed row); a COLLAPSED row keeps its panels nested. So for
# each row: pop its nested panels, lay them out, and either flatten (open) or
# re-nest (collapsed).
def reflow(panels):
    y = 0
    out = []
    for p in panels:
        if p["type"] == "row":
            grp = p.pop("panels", [])
            p["gridPos"] = {"h": 1, "w": 24, "x": 0, "y": y}
            y += 1
            if p.get("collapsed"):
                xpos = 0; cur_y = y; row_h = 0
                for g in grp:
                    w = g.get("gridPos", {}).get("w", 24)
                    h = g.get("gridPos", {}).get("h", 4)
                    if xpos + w > 24:
                        xpos = 0; cur_y += row_h; row_h = 0
                    g["gridPos"] = {"h": h, "w": w, "x": xpos, "y": cur_y}
                    xpos += w
                    row_h = max(row_h, h)
                p["panels"] = grp
                out.append(p)
                y = cur_y + row_h  # reserve the (hidden) content height
            else:
                out.append(p)  # open row: panels follow as top-level siblings
                xpos = 0; cur_y = y; row_h = 0
                for g in grp:
                    w = g.get("gridPos", {}).get("w", 24)
                    h = g.get("gridPos", {}).get("h", 8)
                    if xpos + w > 24:
                        xpos = 0; cur_y += row_h; row_h = 0
                    g["gridPos"] = {"h": h, "w": w, "x": xpos, "y": cur_y}
                    out.append(g)
                    xpos += w
                    row_h = max(row_h, h)
                y = cur_y + row_h
        else:
            p["gridPos"] = {"h": p.get("gridPos", {}).get("h", 8),
                            "w": p.get("gridPos", {}).get("w", 24),
                            "x": 0, "y": y}
            y += p["gridPos"]["h"]
            out.append(p)
    return out

panels = reflow(panels)

dash = {
    "title": "SGLang — Qwen3.8-27B NVFP4 (RTX5090)",
    "tags": ["sglang", "qwen3.8-27b", "nvfp4", "dflash2", "dspark", "rtx5090"],
    "timezone": "browser",
    "refresh": "10s",
    "editable": True,
    "schemaVersion": 39,
    "version": 1,
    "uid": "sglang-qwen38-27b",
    "time": {"from": "now-1h", "to": "now"},
    "templating": {"list": []},
    "annotations": {"list": []},
    "panels": panels,
}

out = os.path.join(HERE, "grafana", "dashboards", "sglang-qwen38-27b.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(dash, f, indent=2)
print(f"wrote {out} ({os.path.getsize(out)} bytes, {len(panels)} top-level items)")
print("valid JSON:", bool(json.load(open(out))))
