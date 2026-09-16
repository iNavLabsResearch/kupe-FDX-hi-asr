#!/usr/bin/env python3
"""Generate the paper's figures (matplotlib -> PNG). Clean edge-to-edge arrows, no overlap.
English labels only (Devanagari samples are typeset by LaTeX). All numbers come from the
code/configs so the figures match the implementation exactly."""
import os
import sys
from collections import namedtuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from kupefdx.fcgen.scenarios import distribution_table  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)
C = {"enc": "#cfe8ff", "head": "#ffe0b3", "lm": "#d7f5d0", "out": "#f6d6e6",
     "data": "#e8e2ff", "quiet": "#eaffea", "edge": "#333333"}

Box = namedtuple("Box", "x y w h")


def draw(ax, b, text, color, fs=9):
    ax.add_patch(FancyBboxPatch((b.x, b.y), b.w, b.h,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec=C["edge"], lw=1.2))
    ax.text(b.x + b.w / 2, b.y + b.h / 2, text, ha="center", va="center", fontsize=fs)
    return b


def side(b, s):
    return {"l": (b.x, b.y + b.h / 2), "r": (b.x + b.w, b.y + b.h / 2),
            "t": (b.x + b.w / 2, b.y + b.h), "b": (b.x + b.w / 2, b.y)}[s]


def connect(ax, b1, s1, b2, s2, style="-|>", rad=0.0, color=None, dashed=False):
    p1, p2 = side(b1, s1), side(b2, s2)
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=13, lw=1.3,
                                 color=color or C["edge"], shrinkA=1, shrinkB=1,
                                 connectionstyle=f"arc3,rad={rad}",
                                 linestyle="--" if dashed else "-"))


def fig_arch():
    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.set_xlim(0, 24); ax.set_ylim(0, 13); ax.axis("off")
    inp = draw(ax, Box(0.4, 5.6, 2.8, 1.8), "16 kHz mic\n320–640 ms\nchunks", C["data"], 8.5)
    den = draw(ax, Box(3.9, 5.6, 2.6, 1.8), "RNNoise\ndenoise", C["data"])
    enc = draw(ax, Box(7.1, 5.0, 3.3, 3.0), "omniASR_W2V\n(SSL wav2vec2)\nblock-causal\nstreaming mask", C["enc"], 9)
    ctc = draw(ax, Box(11.4, 10.4, 4.2, 1.9), "CTC head (linear)\nDevanagari chars\n(alignment · fast path · timing)", C["head"], 7.8)
    prj = draw(ax, Box(11.4, 7.7, 4.2, 1.8), "Projector (MLP)\n→ 832-d soft prompts", C["head"], 8.3)
    qnt = draw(ax, Box(11.4, 5.0, 4.2, 1.8), "k-means quantizer\n(optional) → <aud_k>", C["head"], 8.3)
    fch = draw(ax, Box(11.4, 2.3, 4.2, 1.8), "FloorControlHead\nper-chunk 5-class", C["head"], 8.3)
    ctx = draw(ax, Box(17.4, 10.4, 4.2, 1.9), "domain tag +\nconversation context\n<hist> … </hist>", C["data"], 8)
    lm = draw(ax, Box(17.4, 5.6, 4.2, 3.4), "Nandi-Mini-150M\nhidden 832\nvocab 131072\nfull finetune", C["lm"], 9.5)
    out = draw(ax, Box(17.4, 1.6, 4.2, 2.8), "corrected transcript\n+ floor-control\n(BC / THINK /\nEOS / SILENCE)", C["out"], 8.3)

    connect(ax, inp, "r", den, "l")
    connect(ax, den, "r", enc, "l")
    for h, r in ((ctc, 0.12), (prj, 0.04), (qnt, -0.04), (fch, -0.14)):
        connect(ax, enc, "r", h, "l", rad=r)
    connect(ax, prj, "r", lm, "l", rad=0.05)
    connect(ax, qnt, "r", lm, "l", rad=-0.05)
    connect(ax, ctx, "b", lm, "t")
    connect(ax, lm, "b", out, "t")
    ax.set_title("Figure 1. KupeFDX-hi-asr architecture (streaming, single forward path)",
                 fontsize=12, loc="left")
    fig.savefig(os.path.join(OUT, "fig_arch.png"), dpi=170, bbox_inches="tight"); plt.close(fig)


def fig_seq():
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.set_xlim(0, 25.5); ax.set_ylim(0, 3.2); ax.axis("off")
    toks = [("<hist> dom ctx </hist>", C["data"], 3.6), ("bos", C["lm"], 1.0),
            ("<audio>", C["head"], 1.5), ("proj(a₀..a_L)", C["enc"], 3.4),
            ("</audio>", C["head"], 1.5), ("<aud_k>*", C["head"], 1.5),
            ("y₀ … y_m  (+ inline flags)", C["out"], 4.6), ("<eos>", C["lm"], 1.0)]
    x, xs = 0.3, []
    for t, c, w in toks:
        draw(ax, Box(x, 1.2, w, 1.1, ), t, c, 8)
        ax.text(x + w / 2, 1.75, t, ha="center", va="center", fontsize=8)  # ensure text on top
        xs.append((x, x + w))
        x += w + 0.28
    pre_end = xs[5][1]
    ax.annotate("", (0.3, 0.85), (pre_end, 0.85), arrowprops=dict(arrowstyle="-", color="#999"))
    ax.text((0.3 + pre_end) / 2, 0.45, "prefix  (label = −100)", ha="center", fontsize=8, color="#666")
    ax.annotate("", (xs[6][0], 0.85), (xs[7][1], 0.85), arrowprops=dict(arrowstyle="-", color="#999"))
    ax.text((xs[6][0] + xs[7][1]) / 2, 0.45, "supervised target", ha="center", fontsize=8, color="#666")
    ax.set_title("Figure 2. Sequence fed to Nandi as inputs_embeds (text vocab reused, not resized)",
                 fontsize=12, loc="left")
    fig.savefig(os.path.join(OUT, "fig_seq.png"), dpi=170, bbox_inches="tight"); plt.close(fig)


def fig_stream():
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.set_xlim(0, 24); ax.set_ylim(0, 7.2); ax.axis("off")
    ch = draw(ax, Box(0.4, 3.0, 2.9, 1.6), "audio chunk\n(320–640 ms)", C["data"], 8.5)
    en = draw(ax, Box(4.0, 3.0, 3.0, 1.6), "encoder\n(rolling buffer)", C["enc"], 8.5)
    ctc = draw(ax, Box(7.5, 5.0, 3.9, 1.5), "CTC: timing +\noptional draft\n(NOT the transcript)", C["head"], 7.8)
    fh = draw(ax, Box(7.5, 1.0, 3.9, 1.5), "FloorControlHead\nsoftmax(5)", C["head"], 8.3)
    dec = draw(ax, Box(12.1, 0.8, 4.6, 1.9), "StreamDecider\ntemperature · bias ·\nthreshold · hysteresis\n· EOS latch", C["lm"], 8)
    rec = draw(ax, Box(17.6, 4.2, 5.8, 2.3), "per-chunk record:\ncorrected transcript = NANDI,\nbackchannel, thinking,\neos_flag, silence_flag", C["out"], 8)
    quiet = draw(ax, Box(17.6, 1.0, 5.8, 2.0), "DEFAULT = NOTHING\n(all signal fields null/false)\n— most chunks —", C["quiet"], 8.5)
    connect(ax, ch, "r", en, "l")
    connect(ax, en, "r", ctc, "l", rad=0.12)
    connect(ax, en, "r", fh, "l", rad=-0.12)
    connect(ax, fh, "r", dec, "l")
    connect(ax, ctc, "r", rec, "l", rad=0.05)
    connect(ax, dec, "r", quiet, "l", rad=-0.05)
    connect(ax, dec, "t", rec, "b", rad=-0.2, color="#c04d84")
    ax.text(15.2, 3.55, "fire (rare)", fontsize=7, color="#c04d84")
    ax.set_title("Figure 3. Streaming decision — controllable, and quiet by default",
                 fontsize=12, loc="left")
    fig.savefig(os.path.join(OUT, "fig_stream.png"), dpi=170, bbox_inches="tight"); plt.close(fig)


def fig_dist():
    rows = distribution_table()
    names = [r[0] for r in rows]; pct = [r[1] for r in rows]
    noflag = {"nothing_happens", "midsentence_pause", "false_trigger_trap", "barge_in"}
    colors = ["#8fbf8f" if n in noflag else "#d98fb0" for n in names]
    fig, ax = plt.subplots(figsize=(9, 4.0))
    order = sorted(range(len(names)), key=lambda i: pct[i])
    names = [names[i] for i in order]; pct = [pct[i] for i in order]; colors = [colors[i] for i in order]
    ax.barh(names, pct, color=colors, ec="#333", lw=0.6)
    for i, p in enumerate(pct):
        ax.text(p + 0.3, i, f"{p}%", va="center", fontsize=8)
    ax.set_xlabel("share of generated floor-control rows (%)"); ax.set_xlim(0, 24)
    noflag_pct = sum(p for n, p in zip(names, pct) if n in noflag)
    ax.set_title(f"Figure 4. Scenario distribution (config-driven) — green = no-flag ≈ {noflag_pct:.0f}%",
                 fontsize=11.5, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_dist.png"), dpi=170, bbox_inches="tight"); plt.close(fig)


def fig_hours():
    fig, ax = plt.subplots(figsize=(8.5, 2.3))
    segs = [("core clean ASR", 2500, "#7fa8d8"), ("domain packs", 400, "#e0a35a"),
            ("floor-control", 300, "#8fbf8f")]
    left = 0
    for name, h, c in segs:
        ax.barh([0], [h], left=left, color=c, ec="#333", label=f"{name} ({h} h)")
        ax.text(left + h / 2, 0, f"{h} h", ha="center", va="center", fontsize=9)
        left += h
    ax.set_yticks([]); ax.set_xlabel("hours"); ax.set_xlim(0, 3300)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.4), ncol=3, fontsize=8, frameon=False)
    ax.set_title("Figure 5. Training-data budget ≈ 3,200 h (target for <5% streaming WER)",
                 fontsize=11.5, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_hours.png"), dpi=170, bbox_inches="tight"); plt.close(fig)


def fig_phases():
    import numpy as np
    phases = ["P1 CTC", "P2 Align", "P3 Joint", "P4 FloorCtrl", "P5 Domain"]
    modules = ["encoder", "CTC head", "projector", "Nandi", "FC head"]
    M = np.array([[1, 0, 1, 0, 0], [1, 0, 1, 0, 0], [0, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1], [0, 0, 1, 1, 0]]).T
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.imshow(M, cmap="Greens", vmin=0, vmax=1.7, aspect="auto")
    ax.set_xticks(range(len(phases))); ax.set_xticklabels(phases, fontsize=9)
    ax.set_yticks(range(len(modules))); ax.set_yticklabels(modules, fontsize=9)
    for i in range(len(modules)):
        for j in range(len(phases)):
            ax.text(j, i, "train" if M[i, j] else "frozen", ha="center", va="center",
                    fontsize=7.5, color="#123" if M[i, j] else "#999")
    weights = ["ctc=1", "lm+ctc·.3", "lm+ctc·.3\nfcc·.2", "lm+ctc·.1\nfc·3+fcc·1", "lm+ctc·.1"]
    ax.set_xticks(range(len(phases)))
    for j, w in enumerate(weights):
        ax.text(j, len(modules) - 0.42, w, ha="center", va="top", fontsize=6.4, color="#555")
    ax.set_title("Figure 6. Five training phases — what trains (green) and loss weights",
                 fontsize=11.5, loc="left")
    ax.set_ylim(len(modules) - 0.5 + 0.6, -0.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_phases.png"), dpi=170, bbox_inches="tight"); plt.close(fig)


def fig_audioflow():
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.set_xlim(0, 24); ax.set_ylim(0, 5); ax.axis("off")
    a = draw(ax, Box(0.3, 1.8, 3.4, 1.6), "640 ms chunk\n16 kHz mono\n= 10,240 samples", C["data"], 8.3)
    b = draw(ax, Box(4.3, 1.8, 3.6, 1.6), "omniASR_W2V\nSSL encoder (causal)\n20 ms/frame", C["enc"], 8.3)
    c = draw(ax, Box(8.5, 1.8, 3.6, 1.6), "32 frames\n× D (continuous)\nSSL only", C["enc"], 8.3)
    d = draw(ax, Box(12.7, 1.8, 3.6, 1.6), "Projector (MLP)\n32 × 832-d\nsoft prompts", C["head"], 8.3)
    e = draw(ax, Box(16.9, 1.5, 3.4, 2.2), "Nandi-Mini\ninputs_embeds:\n<audio> 32 vecs </audio>\n(no discretization)", C["lm"], 8)
    for x, y in ((a, b), (b, c), (c, d), (d, e)):
        connect(ax, x, "r", y, "l")
    ax.text(0.3, 4.2, "Per chunk: 640 ms → 32 continuous 832-d vectors (FastConformer: 8). We use "
            "ONLY the SSL encoder weights — our own CTC head is trained fresh. Streaming buffer + "
            "block-causal mask; left context = 2 chunks.",
            fontsize=8.5, color="#333")
    ax.set_title("Figure 7. Audio → SLM: chunk sizes and how much is passed (continuous, per chunk)",
                 fontsize=12, loc="left")
    fig.savefig(os.path.join(OUT, "fig_audioflow.png"), dpi=170, bbox_inches="tight"); plt.close(fig)


def fig_agentflow():
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.set_xlim(0, 24); ax.set_ylim(0, 11); ax.axis("off")
    # inputs (static = green tint, generated = pink tint)
    wav = draw(ax, Box(0.3, 8.4, 3.6, 1.9), "real clip WAV\n(STATIC)", "#dff0df", 8.5)
    prb = draw(ax, Box(0.3, 5.7, 3.6, 1.9), "audio_probe →\naudio card\n(pauses, dur, dB)\nMEASURED", C["data"], 7.8)
    tr = draw(ax, Box(0.3, 3.0, 3.6, 1.9), "real transcript\nDevanagari (STATIC)", "#dff0df", 8.3)
    agent = draw(ax, Box(4.7, 5.0, 4.0, 3.2), "LLM agent\n(gpt-5.6-luna)\n~22 rows / hit\nconc. 10 · tqdm", C["out"], 8.5)
    gen = draw(ax, Box(9.4, 5.0, 4.2, 3.2), "GENERATED (agent):\nscenario · context\nflag placements\n+ surfaces\n(timeline)", "#ffdfe8", 8)
    merge = draw(ax, Box(14.3, 5.2, 4.0, 2.8), "merge + validate\n(schema, semantic\nordering, Devanagari)\n→ target_sequence", C["head"], 8)
    row = draw(ax, Box(19.0, 5.4, 4.6, 2.4), "training row (JSONL):\naudio + target_sequence\n+ context + timeline", C["lm"], 8)
    coll = draw(ax, Box(9.4, 1.0, 8.0, 2.4), "collate → model inputs_embeds:\n[ctx] bos <audio> feats </audio> [target + inline flags] eos\n+ per-chunk FC labels (mostly NOTHING)", C["enc"], 8)
    connect(ax, wav, "r", agent, "l", rad=0.12)
    connect(ax, prb, "r", agent, "l")
    connect(ax, tr, "r", agent, "l", rad=-0.12)
    connect(ax, agent, "r", gen, "l")
    connect(ax, gen, "r", merge, "l")
    connect(ax, tr, "b", merge, "b", rad=-0.35, color="#4a8", dashed=True)  # transcript flows in statically
    connect(ax, merge, "r", row, "l")
    connect(ax, row, "b", coll, "r", rad=0.2)
    ax.text(0.3, 10.6, "STATIC (real data): audio + Devanagari transcript.   "
            "GENERATED (agent): flags · surfaces · context.   Token-saving: transcript is NOT re-emitted.",
            fontsize=8.5, color="#333")
    ax.set_title("Figure 8. Agent input → output → merge → what the model trains on",
                 fontsize=12, loc="left")
    fig.savefig(os.path.join(OUT, "fig_agentflow.png"), dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fig_arch(); fig_seq(); fig_stream(); fig_dist(); fig_hours(); fig_phases()
    fig_audioflow(); fig_agentflow()
    print("figures ->", OUT)
