"""KupeFDXModel — omniASR_W2V (causal) -> [CTC head] + [projector -> Nandi].

Sequence fed to Nandi (as inputs_embeds; Nandi's text vocab is reused, extended only
by our control/audio-code tokens):

  [bos] [audio_bos] proj(a_0..a_{L-1}) [audio_eos] (aud_code embeds?) w_0..w_{m-1} [eos]
  |------------------------- prefix (label -100) ----------------------|--- supervised ---|

Loss = lm_weight * LM-CE  +  ctc_weight * CTC(feats)  +  fc_weight * (extra CE on
floor-control target positions). Phase 1 sets lm_weight=0 (CTC-only encoder warmup);
Phase 3 turns everything on. Freeze helpers let train.py stage what trains per phase.
"""
from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import FC_TOKENS
from .ctc_head import CTCHead
from .decoders import _DTYPES, load_decoder
from .encoders import build_encoder
from .env import log
from .frontend import FeatureFrontend
from .text import CharTokenizer
from .tokens import extend_vocab
from .vocab import build_special_tokens


class KupeFDXModel(nn.Module):
    def __init__(self, encoder, ctc_head, frontend, decoder, tokenizer, *,
                 bos_id, eos_id, pad_id, special_ids, n_codes, char_tok,
                 fc_head=None, chunk_frames=32, fc_class_weights=None):
        super().__init__()
        self.encoder = encoder
        self.ctc = ctc_head
        self.frontend = frontend
        self.lm = decoder
        self.tok = tokenizer
        self.char_tok = char_tok
        self.fc_head = fc_head                    # per-chunk stability head
        self.chunk_frames = int(chunk_frames)
        self.fc_class_weights = fc_class_weights or [0.3, 2.0, 2.0, 2.0, 2.0]
        self.bos_id, self.eos_id, self.pad_id = int(bos_id), int(eos_id), int(pad_id)
        self.special_ids = special_ids          # {token_str: id}
        self.n_codes = int(n_codes)
        self.fc_ids = {special_ids[t] for t in FC_TOKENS if t in special_ids}
        # map raw audio-code k (0..n_codes-1) -> its <aud_k> vocab id, for embedding.
        if int(n_codes) > 0:
            c2v = torch.tensor([special_ids[f"<aud_{k}>"] for k in range(int(n_codes))],
                               dtype=torch.long)
            self.register_buffer("code2vocab", c2v, persistent=False)
        else:
            self.code2vocab = None
        self.config = decoder.config

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, cfg):
        dtype = _DTYPES.get(getattr(cfg.base, "dtype", "float32"), torch.float32)
        encoder = build_encoder(cfg, dtype)
        decoder, tok = load_decoder(cfg, dtype)
        lang = getattr(cfg, "lang", "en")
        char_tok = CharTokenizer(lang=lang)
        n_codes = int(getattr(cfg.audio, "n_codes", 0))

        specials = build_special_tokens(n_codes)
        special_ids = extend_vocab(decoder, tok, specials)

        hidden = int(decoder.config.hidden_size)
        enc_dim = int(encoder.out_dim)
        frontend = FeatureFrontend(enc_dim, hidden,
                                   getattr(cfg.audio, "projector", "mlp"),
                                   float(getattr(cfg.audio, "proj_dropout", 0.0))).to(dtype)
        ctc = CTCHead(enc_dim, char_tok.vocab_size, blank=char_tok.blank).to(dtype)
        from .floorcontrol import FC_CLASSES, FloorControlHead
        fc_head = FloorControlHead(enc_dim, n_classes=len(FC_CLASSES)).to(dtype)
        weights = list(getattr(cfg.train, "fc_class_weights", []) or [0.3, 2.0, 2.0, 2.0, 2.0])
        log.info("KupeFDX built | enc_dim=%d hidden=%d ctc_vocab=%d n_codes=%d fc_classes=%d",
                 enc_dim, hidden, char_tok.vocab_size, n_codes, len(FC_CLASSES))
        return cls(encoder, ctc, frontend, decoder, tok,
                   bos_id=tok.bos_token_id, eos_id=tok.eos_token_id,
                   pad_id=tok.pad_token_id, special_ids=special_ids,
                   n_codes=n_codes, char_tok=char_tok, fc_head=fc_head,
                   chunk_frames=int(getattr(cfg.audio, "chunk_frames", 32)),
                   fc_class_weights=weights)

    # ------------------------------------------------------------------ helpers
    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def _dtype(self):
        return self.frontend.audio_bos.dtype

    def _embed_ids(self, ids):
        return self.lm.get_input_embeddings()(ids)      # -> [.., hidden]

    def _encode(self, wave, wave_len, feats, num_frames):
        """Return (feats[B,T,D], flen[B]). Use cached feats if given, else run encoder."""
        if feats is not None:
            return feats.to(self.device).to(self._dtype), num_frames.to(self.device)
        f, flen = self.encoder.features(wave.to(self.device), wave_len.to(self.device))
        return f.to(self._dtype), flen

    def _audio_prefix_embeds(self, feats, flen, code_ids=None, ctx_ids=None):
        """[ (ctx), bos, audio_bos, proj(frames)[:L], audio_eos, (code embeds) ] per sample.
        `ctx_ids` is an optional per-sample list of 1D id tensors (domain tag + history)."""
        proj = self.frontend(feats)                     # [B,T,hidden]
        bos = self._embed_ids(torch.tensor([self.bos_id], device=self.device))[0]
        abos, aeos = self.frontend.audio_bos, self.frontend.audio_eos
        outs, lens = [], []
        for b in range(proj.shape[0]):
            la = int(flen[b].item())
            parts = []
            if ctx_ids is not None and ctx_ids[b] is not None and len(ctx_ids[b]) > 0:
                parts.append(self._embed_ids(ctx_ids[b].to(self.device)))
            parts += [bos[None], abos[None], proj[b, :la], aeos[None]]
            if code_ids is not None and self.n_codes > 0 and self.code2vocab is not None:
                vocab_ids = self.code2vocab.to(self.device)[code_ids[b].clamp(0, self.n_codes - 1)]
                parts.append(self._embed_ids(vocab_ids))
            seq = torch.cat(parts, dim=0)
            outs.append(seq)
            lens.append(seq.shape[0])
        return outs, lens

    # ------------------------------------------------------------------ forward
    def forward(self, *, wave=None, wave_len=None, feats=None, num_frames=None,
                text_ids=None, text_lengths=None, labels=None,
                ctc_targets=None, ctc_target_lens=None, code_ids=None, ctx_ids=None,
                fc_events=None, fc_chunk_weight=0.0,
                lm_weight=1.0, ctc_weight=0.3, fc_weight=1.0, **_):
        feats, flen = self._encode(wave, wave_len, feats, num_frames)
        out = {}
        total = feats.new_zeros(())

        # ---- per-chunk floor-control head (STABILITY): default NOTHING dominates ----
        if fc_chunk_weight > 0 and self.fc_head is not None and fc_events is not None:
            logits = self.fc_head(feats.float(), flen, self.chunk_frames)
            fcl = self.fc_head.loss(logits, fc_events, self.fc_class_weights, self.device)
            if fcl is not None:
                out["fc_chunk_loss"] = fcl.detach()
                total = total + fc_chunk_weight * fcl

        if ctc_weight > 0 and ctc_targets is not None:
            ctc_loss = self.ctc.loss(feats.float(), flen, ctc_targets.to(self.device),
                                     ctc_target_lens.to(self.device))
            out["ctc_loss"] = ctc_loss.detach()
            total = total + ctc_weight * ctc_loss

        if lm_weight > 0 and text_ids is not None:
            code_ids = code_ids.to(self.device) if code_ids is not None else None
            pre, prelens = self._audio_prefix_embeds(feats, flen, code_ids, ctx_ids)
            text_ids = text_ids.to(self.device)
            text_emb = self._embed_ids(text_ids)
            B = feats.shape[0]
            seqs, labs, lens = [], [], []
            for b in range(B):
                m = int(text_lengths[b].item())
                seq = torch.cat([pre[b], text_emb[b, :m]], dim=0)
                seqs.append(seq)
                lens.append(seq.shape[0])
                p = prelens[b]
                lab = torch.cat([torch.full((p,), -100, dtype=torch.long, device=self.device),
                                 labels[b, :m].to(self.device)], dim=0)
                labs.append(lab)
            Lmax = max(lens)
            H = feats.new_zeros(B, Lmax, self.frontend.embed_dim)
            attn = torch.zeros(B, Lmax, dtype=torch.long, device=self.device)
            lab_pad = torch.full((B, Lmax), -100, dtype=torch.long, device=self.device)
            for b in range(B):
                L = lens[b]
                H[b, :L] = seqs[b].to(self._dtype)
                attn[b, :L] = 1
                lab_pad[b, : labs[b].shape[0]] = labs[b]
            res = self.lm(inputs_embeds=H, attention_mask=attn, labels=lab_pad)
            lm_loss = res.loss
            out["lm_loss"] = lm_loss.detach()
            total = total + lm_weight * lm_loss

            if fc_weight and self.fc_ids and hasattr(res, "logits"):
                extra = self._fc_extra_loss(res.logits, lab_pad)
                if extra is not None:
                    out["fc_loss"] = extra.detach()
                    total = total + fc_weight * extra

        out["loss"] = total
        return out

    def _fc_extra_loss(self, logits, labels):
        """Up-weight CE on positions whose target is a floor-control token, so rare
        signal events are learned without letting <NOP> negatives drown them out."""
        sl = logits[:, :-1, :].contiguous()
        tl = labels[:, 1:].contiguous()
        fc_mask = torch.zeros_like(tl, dtype=torch.bool)
        for fid in self.fc_ids:
            fc_mask |= tl == fid
        if fc_mask.sum() == 0:
            return None
        ce = F.cross_entropy(sl.view(-1, sl.shape[-1]), tl.view(-1),
                             ignore_index=-100, reduction="none").view(tl.shape)
        return ce[fc_mask].mean()

    # ------------------------------------------------------------------ decode
    @torch.no_grad()
    def generate(self, *, wave=None, wave_len=None, feats=None, num_frames=None,
                 code_ids=None, ctx_ids=None, max_new_tokens=128):
        feats, flen = self._encode(wave, wave_len, feats, num_frames)
        pre, prelens = self._audio_prefix_embeds(
            feats, flen, code_ids.to(self.device) if code_ids is not None else None, ctx_ids)
        results = []
        for b in range(feats.shape[0]):
            H = pre[b][None].to(self._dtype)            # [1, L, hidden]
            out_ids = []
            for _ in range(int(max_new_tokens)):
                attn = torch.ones(1, H.shape[1], dtype=torch.long, device=self.device)
                logits = self.lm(inputs_embeds=H, attention_mask=attn).logits[:, -1]
                nxt = int(logits.argmax(-1).item())
                if nxt == self.eos_id:
                    break
                out_ids.append(nxt)
                nemb = self._embed_ids(torch.tensor([[nxt]], device=self.device)).to(self._dtype)
                H = torch.cat([H, nemb], dim=1)
            results.append(out_ids)
        return results

    @torch.no_grad()
    def transcribe(self, results_ids):
        return [self.tok.decode(ids, skip_special_tokens=True) for ids in results_ids]

    @torch.no_grad()
    def transcribe_audio(self, wave=None, wave_len=None, feats=None, num_frames=None,
                         max_new_tokens=256):
        """AUTHORITATIVE transcript: Nandi decodes it. Use this for the real output."""
        gen = self.generate(wave=wave, wave_len=wave_len, feats=feats,
                            num_frames=num_frames, max_new_tokens=max_new_tokens)
        return self.transcribe(gen)

    @torch.no_grad()
    def ctc_transcribe(self, wave=None, wave_len=None, feats=None, num_frames=None):
        """NON-AUTHORITATIVE low-latency DRAFT from the CTC head (used only as a fast
        partial while Nandi catches up). The real transcript is transcribe_audio()."""
        feats, _ = self._encode(wave, wave_len, feats, num_frames)
        seqs, _ = self.ctc.greedy(feats.float())
        return [self.char_tok.decode(s) for s in seqs]

    @torch.no_grad()
    def chunk_logits(self, wave=None, wave_len=None, feats=None, num_frames=None):
        """Per-chunk floor-control class LOGITS for the streaming decider (temperature +
        bias are applied there). Returns [n_chunks, n_classes] for a single clip."""
        feats, flen = self._encode(wave, wave_len, feats, num_frames)
        return self.fc_head(feats.float(), flen, self.chunk_frames)[0]

    @torch.no_grad()
    def chunk_probs(self, wave=None, wave_len=None, feats=None, num_frames=None):
        return F.softmax(self.chunk_logits(wave, wave_len, feats, num_frames), dim=-1)

    # ------------------------------------------------------------------ freezing
    def _set(self, module, trainable):
        for p in module.parameters():
            p.requires_grad = bool(trainable)

    def stage(self, *, encoder, ctc, projector, decoder, fc=False):
        self._set(self.encoder, encoder)
        self._set(self.ctc, ctc)
        self._set(self.frontend, projector)
        self._set(self.lm, decoder)
        if self.fc_head is not None:
            self._set(self.fc_head, fc)
        log.info("stage | encoder=%s ctc=%s projector=%s decoder=%s fc_head=%s",
                 encoder, ctc, projector, decoder, fc)

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------ save/load
    def save(self, out_dir, cfg):
        os.makedirs(out_dir, exist_ok=True)
        parts = {"encoder": self.encoder.state_dict(),
                 "ctc": self.ctc.state_dict(),
                 "frontend": self.frontend.state_dict()}
        if self.fc_head is not None:
            parts["fc_head"] = self.fc_head.state_dict()
        torch.save(parts, os.path.join(out_dir, "kupefdx_parts.pt"))
        if hasattr(self.lm, "save_pretrained"):
            self.lm.save_pretrained(os.path.join(out_dir, "kupe-lm"))
        else:
            torch.save(self.lm.state_dict(), os.path.join(out_dir, "tiny_lm.pt"))
        if hasattr(self.tok, "save_pretrained"):
            self.tok.save_pretrained(os.path.join(out_dir, "kupe-lm"))
        meta = {"backend": cfg.backend, "n_codes": self.n_codes,
                "enc_dim": self.frontend.enc_dim, "hidden": self.frontend.embed_dim,
                "bos_id": self.bos_id, "eos_id": self.eos_id, "pad_id": self.pad_id,
                "special_ids": self.special_ids, "encoder_id": cfg.base.encoder_id,
                "decoder_id": cfg.base.decoder_id}
        with open(os.path.join(out_dir, "kupefdx_config.json"), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        log.info("saved KupeFDX -> %s", out_dir)
