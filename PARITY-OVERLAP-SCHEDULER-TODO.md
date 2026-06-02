# Parity TODO — re-enable the OVERLAP SCHEDULER (high-priority lever)

> Branch `perf/nemotron-h-allreduce-fusion`. Added 2026-06-01. Companion to the allreduce+rmsnorm
> fusion already on this branch (commit 2dafea956) and the parity memory
> `~/.claude/.../memory/sglang-vllm-perf-parity.md`.

## Why this is likely the biggest parity lever

Profiling showed the SGLang↔vLLM gap on Nemotron-Ultra-v3 NVFP4 (AA TP4) is **comm- + overhead-bound**,
with the GPU only ~54% busy → **~46% idle on CPU orchestration**. That idle exists because this model
runs `mamba_scheduler_strategy=no_buffer` (the `auto` default), and SGLang **disables the overlap
scheduler under `no_buffer` + radix cache** (`server_args.py:2544-2548` →
`self.disable_overlap_schedule = True`; the in-place Mamba SSM state can't be safely double-buffered).

The overlap scheduler is SGLang's general analog of vLLM's `--async-scheduling` (runs next-step CPU prep
concurrently with the GPU forward). Re-enabling it attacks the 46% idle **directly** — plausibly a bigger
win than the allreduce+rmsnorm fusion (which was a wash at conc=1 because the GPU was already idle-bound).

## What to test (two ways to regain overlap)

1. **`--mamba-scheduler-strategy extra_buffer`** — double-buffers Mamba state (the `mamba_track`
   ping-pong buffer) so overlap works WITH radix cache intact.
   - Requires the model to advertise `support_mamba_cache_extra_buffer` (verify NemotronH does).
   - Constraints: FLA backend; `page_size`/`mamba_track_interval` divisibility; if MTP on,
     `mamba_track_interval >= speculative_num_draft_tokens`. Costs extra memory.
2. **`no_buffer` + `--disable-radix-cache`** — the code's own escape hatch (server_args.py:2514-2515:
   "Overlap scheduling is already supported with no_buffer + disable_radix_cache"). Overlap works, but
   loses radix prefix reuse.

## How to run (oci-hsg, TP4 no-EP, NVFP4 — the AA parity config)

Add the knob via the sbatch (e.g. `SGL_EXTRA_ARGS="--mamba-scheduler-strategy extra_buffer"` or
`SGL_EXTRA_ARGS="--mamba-scheduler-strategy no_buffer --disable-radix-cache"`), TP4 single node:
```
SGLANG_SRC=<this fork> SGLANG_ENABLE_SPEC_V2=1 \
QUANTIZATION=modelopt_fp4 KV_CACHE_DTYPE=fp8_e4m3 MEM_FRACTION=0.90 \
TP_SIZE=4 EP_SIZE=1 MTP=1 SPEC_NUM_STEPS=4 SPEC_TOPK=1 SPEC_NUM_DRAFT_TOKENS=5 \
SGL_EXTRA_ARGS="--mamba-scheduler-strategy extra_buffer" \
sbatch --exclusive --nodes=1 deployment/sbatch/sglang_multinode_oci_hsg.sbatch
```
Confirm in the server log that overlap is NOT disabled (no "Disabling overlap schedule" line).

## What to measure

- **Default workload 10k ISL / 2k OSL**, conc 1/2/4/8 (`deployment/aiperf_benchmarks/run_aiperf_benchmarks.sh`,
  WORKLOAD=aa). Realistic acceptance (~70%), NOT 16k OSL (looping inflates MTP acceptance → flatters vLLM).
- Compare tok/s/GPU vs the current SGLang numbers and vLLM. Re-profile a decode step (`analyze_trace.py`)
  to confirm GPU-busy % rises (idle shrinks).
- Test both no-MTP and MTP (spec-v2 on). Watch memory headroom with `extra_buffer`.

## Sequencing note

This is queued **after** the DEP8 DP-attention min-1 / MAX_LEN+mamba-trim work on
`feat/dp-attention-min1-lockstep` resolves (works or not), per 2026-06-01 plan.

## RESULT (2026-06-01) — overlap scheduler is a MARGINAL lever, not the big win

- `extra_buffer` is **unsupported** for `NemotronHForCausalLM` (hard assert) — ruled out.
- `no_buffer` + `--disable-radix-cache` **does** enable the overlap scheduler (TP4, confirmed: no
  "Disabling overlap schedule" log).
- **Measured (TP4 no-EP NVFP4, 10k/2k, NO MTP, overlap ON vs the radix-cache/overlap-OFF baseline):**
  tok/s/GPU c1/2/4/8 = 29.5/53.4/91.5/152.8  vs baseline 28.9/51.7/88.2/149.8 → **only ~+2-3%.**
- **Key realization:** `server_args.py:2543` disables overlap ONLY when `speculative_algorithm is None`.
  So **MTP runs already have the overlap scheduler ON** (MTP forces `--disable-radix-cache`). The ~46%
  GPU-idle was measured on the MTP path, which already had overlap — so the idle is **spec
  orchestration** (addressed by **spec-v2, +21%**, already done), NOT the general overlap scheduler.
- **Conclusion:** overlap-scheduler-via-disable-radix is a marginal no-MTP lever and a no-op for MTP.
  Stream-2 perf work should refocus on the **allreduce+rmsnorm FUSION at concurrency** (this branch's
  actual contribution; only tested at conc=1 = wash; expected to help at conc>=2) and spec-v2.
