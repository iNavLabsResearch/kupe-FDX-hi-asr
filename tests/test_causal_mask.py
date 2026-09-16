"""Block-causal streaming mask must not leak the future: a frame's output cannot
change when frames strictly to its right (beyond its allowed left context) change.
This is what makes the validated WER a real streaming number."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch

from kupefdx.encoders import TinyEncoder, block_causal_mask


def test_mask_shape_and_triangularity():
    m = block_causal_mask(10, chunk_frames=2, left_chunks=0, device="cpu", dtype=torch.float32)
    # within-chunk allowed, future chunks blocked, chunks older than left_chunks blocked.
    assert torch.isinf(m[0, 2])          # frame 0 (chunk0) cannot see frame 2 (chunk1)
    assert m[2, 2] == 0                  # can see own frame
    assert torch.isinf(m[2, 0])          # left_chunks=0 -> cannot see previous chunk
    assert m[3, 2] == 0                  # same chunk (chunk1) visible both ways


def test_no_future_leakage():
    torch.manual_seed(0)
    enc = TinyEncoder(out_dim=32, layers=2, chunk_frames=4, left_chunks=1).eval()
    S = 16000
    wave = torch.randn(1, S)
    with torch.no_grad():
        f1, fl = enc.features(wave, torch.tensor([S]))
        wave2 = wave.clone()
        wave2[:, -3200:] += torch.randn(1, 3200)      # perturb only the last ~200ms
        f2, _ = enc.features(wave2, torch.tensor([S]))
    T = f1.shape[1]
    early = T // 2
    # early frames (well before the perturbed tail) must be unchanged.
    diff = (f1[:, :early] - f2[:, :early]).abs().max().item()
    assert diff < 1e-4, f"future leaked into earlier frames: max diff {diff}"
    print("no-future-leakage: OK (early-frame max diff %.2e)" % diff)


if __name__ == "__main__":
    test_mask_shape_and_triangularity()
    test_no_future_leakage()
    print("ALL CAUSAL-MASK TESTS PASSED")
