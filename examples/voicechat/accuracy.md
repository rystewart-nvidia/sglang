# Accuracy validation

The SGLang VoiceChat stages were validated against the
[vLLM-Omni](https://github.com/vllm-project/vllm-omni) implementation of the same
checkpoint, added in [vllm-omni#5842](https://github.com/vllm-project/vllm-omni/pull/5842),
on a 15.61 s reference sample — 196 acoustic frames on the frame-locked 12.5 Hz
timeline. SGLang runs the thinker in bfloat16; the vLLM-Omni reference runs
float32.

| stage | comparison | result |
|---|---|---|
| Thinker (`NemotronDuplexHForCausalLM`) | frame-locked text timeline, token for token | **196/196 (100%)** |
| Audio to text, end to end | sidecar perception into the thinker, against the same reference timeline | **196/196 (100%)** |

The second row is the deployed path: a WAV in, a text timeline out, through the
sidecar's streaming perception encoder and the SGLang thinker together.

## Perception and codec are NeMo modules on both sides

`nemo_audio_sidecar.py` imports `PerceptionCacheManager` and `RVQVAEModel` from
`nemo.collections.speechlm2`, and the reference implementation uses the same
modules. Comparing them directly measures streaming behaviour rather than model
correctness: the sidecar encodes frame by frame with a cache, while the reference
encodes the whole utterance at once.

That difference is small and deterministic — cosine similarity 0.99943, maximum
absolute difference 2.8e-02, identical with the perception CUDA graph enabled or
disabled — and it changes no tokens on this sample.

## The talker is not compared

`EarTTSForCausalLM` samples. MaskGIT runs `num_iter=8` and draws twice per
iteration: a Gumbel mixture selection, and the residual noise added to the
predicted mean. Two implementations therefore consume independent RNG streams in
their own order, so two *correct* implementations still agree on only about 50%
of codes. Measured agreement against the reference is 49.95%, which is the
expected result rather than a defect.

No seed closes that gap. Matching would require both implementations to consume
the RNG identically, at which point the comparison no longer tests two
implementations. Validating the talker against a reference would instead mean
comparing pre-sampling distributions, teacher-forced on the reference's own
codes, rather than the emitted codes themselves.

## Scope

These are single-sample results on the reference input. They demonstrate exact
agreement on that sample. They are not a claim that the streaming perception path
is numerically equivalent to full-utterance encoding in general.
