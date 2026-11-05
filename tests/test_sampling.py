"""Check that FastGPT's sampling gives the same distribution as Hugging Face's logits processors. Runs on CPU.

    python tests/test_sampling.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

import fast_gpt

V = 1026  # XTTS-v2 audio vocabulary size


def main(trials=200):
    worst = 0.0
    real_multinomial = torch.multinomial
    for trial in range(trials):
        g = torch.Generator().manual_seed(trial)
        logits = torch.randn(1, V, generator=g) * (1 + 3 * torch.rand(1, generator=g))
        ids = torch.randint(0, V, (1, 40), generator=g)
        temp = float(0.3 + torch.rand(1, generator=g) * 1.2)
        top_p = float(0.5 + torch.rand(1, generator=g) * 0.5)
        top_k = int(torch.randint(1, 200, (1,), generator=g))
        pen = float(1 + torch.rand(1, generator=g) * 9)

        expected = LogitsProcessorList([
            RepetitionPenaltyLogitsProcessor(pen),
            TemperatureLogitsWarper(temp),
            TopKLogitsWarper(top_k),
            TopPLogitsWarper(top_p),
        ])(ids, logits.clone()).softmax(-1)

        seen = torch.zeros(V, dtype=torch.bool)
        seen[ids[0]] = True
        state = types.SimpleNamespace(
            seen=seen,
            rep_penalty=torch.tensor(pen),
            temperature=torch.tensor(temp),
            top_p=torch.tensor(top_p),
            vocab=V,
        )
        captured = {}

        def fake_multinomial(p, n):
            captured["p"] = p
            return real_multinomial(p, n)

        torch.multinomial = fake_multinomial
        try:
            fast_gpt.FastGPT._sample(state, logits.clone(), top_k, True)
        finally:
            torch.multinomial = real_multinomial
        worst = max(worst, (captured["p"] - expected).abs().max().item())

    print(f"{trials} random cases, max probability difference vs Hugging Face: {worst:.2e}")
    assert worst < 1e-6, "sampling distribution differs from Hugging Face"


if __name__ == "__main__":
    main()
