"""Frame-locked parity check for the VoiceChat thinker.

Drives NemotronDuplexHForCausalLM over a streaming session and compares the
emitted text timeline against a reference, token for token.

The thinker decodes greedily, so this is a true parity check: its timeline can
be compared against another implementation of the same model and must agree
exactly. Any mismatch is a real regression, and the first divergent frame is
printed with surrounding context.

The talker (EarTTSForCausalLM) is not covered here and cannot be checked this
way: it samples, so two implementations draw from independent RNG streams and
disagree on roughly half the codes even when both are correct.

Acoustic frames are supplied as a saved tensor rather than computed here, so the
test depends only on the thinker: no encoder, no audio stack, no sidecar.

Sampling must stay greedy with ignore_eos. Never set min_tokens -- the
tokenizer's EOS doubles as the PAD/silence token the model emits on silent
frames, so masking it forces speech through the entire utterance.

The timeline is frame-locked at 12.5 Hz, so the reply budget is the input
duration. An input without enough trailing silence truncates the reply
*silently*; this test then reports a length mismatch rather than anything more
obviously diagnostic.

Usage:
    python voicechat_thinker_parity.py \
        --checkpoint /path/to/NVIDIA-NemotronLabs-VoiceChat-11B \
        --thinker-stage /path/to/converted/duplex \
        --acoustic-frames frames.pt \
        --reference-tokens reference_text_tokens.json

No reference is checked in, since these artifacts are large and binary. Pass
--emit without --reference-tokens to run the stage and write its timeline
instead of comparing, so a reference can be captured once from a known-good
commit and used to gate later changes. frames.pt is the perception stage's
output, [N, hidden], and is the only input these tests do not produce
themselves.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

# Must match the prompt the reference implementation was run with. The prompt
# occupies the leading timeline rows, so any difference shifts every frame
# after it and looks like a thinker bug rather than a harness mismatch.
DEFAULT_SYSTEM_PROMPT = (
    "You are an AI voice assistant developed by NVIDIA. "
    "Your name is NVIDIA Voice Chat. "
    "Answer in a spoken, conversational style rather than a written one. "
    "Do not repeat the same sentence over and over again. "
    "Start the conversation by greeting the user."
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--thinker-stage", required=True)
    ap.add_argument(
        "--acoustic-frames",
        required=True,
        help="saved [N, hidden] tensor of encoder output",
    )
    ap.add_argument("--reference-tokens", help="json list of reference text token ids")
    ap.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="must match the prompt the reference was captured with; a "
        "mismatch shifts the whole timeline and reads as a thinker bug",
    )
    ap.add_argument("--emit", help="optional path to write the emitted timeline")
    args = ap.parse_args()
    if args.reference_tokens is None and not args.emit:
        ap.error(
            "pass --reference-tokens to compare against, or --emit to write a new reference"
        )

    cfg = json.loads((pathlib.Path(args.checkpoint) / "config.json").read_text())
    stt = cfg["model"]["stt"]["model"]

    frames = torch.load(
        args.acoustic_frames, map_location="cpu", weights_only=True
    ).float()
    if frames.dim() != 2:
        raise ValueError(
            f"--acoustic-frames must be [N, hidden]; got {tuple(frames.shape)}"
        )
    n_frames = frames.shape[0]

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(stt["pretrained_llm"], trust_remote_code=False)
    bos = tok.convert_tokens_to_ids(stt.get("bos_token", "<s>"))
    eos = tok.convert_tokens_to_ids(stt.get("eos_token", "</s>"))
    pad = tok.convert_tokens_to_ids(stt.get("pad_token", "<SPECIAL_12>"))
    prompt_ids = (
        [bos] + tok.encode(args.system_prompt, add_special_tokens=False) + [eos]
    )
    print(f"frames={n_frames} prompt={len(prompt_ids)} pad={pad}")

    from sglang import Engine

    engine = Engine(
        model_path=args.thinker_stage,
        dtype="bfloat16",
        mem_fraction_static=0.75,
        context_length=8192,
        max_running_requests=2,
        skip_tokenizer_init=True,
        enable_streaming_session=True,
        log_level="warning",
    )
    session = engine.open_session(8192, streaming=True)
    params = {
        "sampling_params": {
            "max_new_tokens": 1,
            "temperature": 0.0,
            "ignore_eos": True,
        },
        "session_params": {"id": session, "rid": None},
    }

    emitted, function_prev = [], pad
    try:
        out = engine.generate(
            input_ids=prompt_ids + [pad],
            custom_inputs={
                "is_initial_prefill": True,
                "prompt_length": len(prompt_ids),
                "acoustic_embedding": frames[0:1].tolist(),
            },
            **params,
        )
        emitted.append(out["output_ids"][0])
        function_prev = out["meta_info"]["function_tokens"][-1]
        for t in range(1, n_frames):
            out = engine.generate(
                input_ids=[],
                custom_inputs={
                    "acoustic_embedding": frames[t : t + 1].tolist(),
                    "input_function_ids": [function_prev],
                },
                **params,
            )
            emitted.append(out["output_ids"][0])
            function_prev = out["meta_info"]["function_tokens"][-1]
    finally:
        engine.close_session(session)
        engine.shutdown()

    if args.emit:
        pathlib.Path(args.emit).write_text(json.dumps(emitted))
        if args.reference_tokens is None:
            print(f"wrote {len(emitted)} tokens to {args.emit}")
            return 0

    reference = json.loads(pathlib.Path(args.reference_tokens).read_text())
    if len(emitted) != len(reference):
        print(f"FAIL: emitted {len(emitted)} tokens, reference has {len(reference)}")
        return 1

    mismatched = [i for i, (a, b) in enumerate(zip(emitted, reference)) if a != b]
    matched = len(reference) - len(mismatched)
    print(
        f"exact match: {matched}/{len(reference)} = "
        f"{100.0 * matched / len(reference):.2f}%"
    )
    if mismatched:
        i = mismatched[0]
        lo, hi = max(0, i - 3), min(len(reference), i + 5)
        print(f"first divergence at frame {i}")
        print(f"  emitted   {emitted[lo:hi]}")
        print(f"  reference {reference[lo:hi]}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
