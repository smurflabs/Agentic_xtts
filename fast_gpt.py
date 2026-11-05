"""CUDA-graph decoder for XTTS-v2's GPT. Drop-in replacement for gpt.generate / gpt.get_generator.

Why: XTTS generates ~21 audio tokens per second of speech, one at a time. With Hugging Face generate() every
token costs hundreds of small kernel launches plus Python overhead, so on a slow CPU the GPU sits idle most of
the time. Here the GPT2 attention memory (KV cache) is allocated once at full length, so a decode step always
has the same shapes; the whole step (embeddings, 30 transformer layers, logits processing, sampling) is
recorded once as a CUDA graph and replayed per token with a single launch.

Use:  fast_gpt.install(xtts_model)   # xtts_model = TTS(...).synthesizer.tts_model, already on CUDA
XTTS's own inference() / inference_stream() then run unchanged on top.
"""
import torch
import torch.nn.functional as F

# Check for the stop token every N steps instead of every step (each check stalls the CPU on the GPU).
STOP_CHECK_EVERY = 8
# One graph per (top_k, do_sample); keep the most recent few.
MAX_GRAPHS = 8


class FastGPT:
    def __init__(self, gpt):
        self.gpt = gpt  # TTS.tts.layers.xtts.gpt.GPT
        hf = gpt.gpt  # transformers GPT2Model (wpe/wte unused)
        self.blocks = hf.h
        self.ln_f = hf.ln_f
        p = next(gpt.parameters())
        self.device, self.dtype = p.device, p.dtype
        self.n_layer = len(self.blocks)
        self.n_head = gpt.heads
        self.d_model = gpt.model_dim
        self.d_head = self.d_model // self.n_head
        self.max_len = gpt.gpt_inference.config.n_positions
        self.max_new = gpt.max_gen_mel_tokens
        self.stop = gpt.stop_audio_token
        self.vocab = gpt.mel_head.out_features

        dev = self.device
        shape = (self.n_layer, 1, self.n_head, self.max_len, self.d_head)
        self.k_cache = torch.zeros(shape, device=dev, dtype=self.dtype)
        self.v_cache = torch.zeros(shape, device=dev, dtype=self.dtype)
        self.arange_len = torch.arange(self.max_len, device=dev)

        # Static step inputs/state (graph reads and updates these in place)
        self.tok = torch.zeros(1, 1, dtype=torch.long, device=dev)  # token being fed
        self.pos = torch.zeros(1, dtype=torch.long, device=dev)  # its absolute cache position
        self.mel_pos = torch.zeros(1, dtype=torch.long, device=dev)  # its index among audio tokens (0 = start token)
        self.seen = torch.zeros(self.vocab, dtype=torch.bool, device=dev)  # for repetition penalty
        self.temperature = torch.ones((), device=dev)
        self.top_p = torch.ones((), device=dev)
        self.rep_penalty = torch.ones((), device=dev)
        # Outputs per step: out_tokens[i] = token sampled after feeding audio token i; latents[i] = its hidden state
        self.out_tokens = torch.zeros(self.max_new + 1, dtype=torch.long, device=dev)
        self.latents = torch.zeros(self.max_new + 1, self.d_model, device=dev, dtype=self.dtype)

        self.graphs: dict[tuple[int, bool], torch.cuda.CUDAGraph] = {}
        self.pool = None

    # ---------- model math ----------

    def _forward(self, x, positions, mask):
        """x: (1, T, C) embeddings at cache `positions` (T,). mask: (T, max_len) bool. Returns normed hidden (1, T, C)."""
        B, T, C = x.shape
        H, D = self.n_head, self.d_head
        for i, blk in enumerate(self.blocks):
            h = blk.ln_1(x)
            q, k, v = blk.attn.c_attn(h).split(C, dim=2)
            q = q.view(B, T, H, D).transpose(1, 2)
            self.k_cache[i].index_copy_(2, positions, k.view(B, T, H, D).transpose(1, 2))
            self.v_cache[i].index_copy_(2, positions, v.view(B, T, H, D).transpose(1, 2))
            a = F.scaled_dot_product_attention(q, self.k_cache[i], self.v_cache[i], attn_mask=mask)
            x = x + blk.attn.c_proj(a.transpose(1, 2).reshape(B, T, C))
            x = x + blk.mlp(blk.ln_2(x))
        return self.gpt.final_norm(self.ln_f(x))

    def _sample(self, logits, top_k: int, do_sample: bool):
        """Same processing order as Hugging Face: repetition penalty, temperature, top-k, top-p, sample."""
        scores = logits.float()
        penalized = torch.where(scores < 0, scores * self.rep_penalty, scores / self.rep_penalty)
        scores = torch.where(self.seen, penalized, scores)
        if not do_sample:
            return scores.argmax(-1, keepdim=True)
        scores = scores / self.temperature
        if 0 < top_k < self.vocab:
            kth = torch.topk(scores, top_k, dim=-1).values[..., -1:]
            scores = scores.masked_fill(scores < kth, float("-inf"))
        sorted_scores, idx = torch.sort(scores, descending=False)
        cum = sorted_scores.softmax(-1).cumsum(-1)
        remove = cum <= (1 - self.top_p)
        remove[..., -1:] = False  # always keep the most likely token
        scores = scores.masked_fill(remove.scatter(-1, idx, remove), float("-inf"))
        return torch.multinomial(scores.softmax(-1), 1)

    def _step(self, top_k, do_sample):
        """One decode step on static buffers. Captured into a CUDA graph."""
        x = self.gpt.mel_embedding(self.tok) + self.gpt.mel_pos_embedding.emb(self.mel_pos)[None]
        mask = (self.arange_len <= self.pos)[None]
        latent = self._forward(x, self.pos, mask)[:, -1]
        nxt = self._sample(self.gpt.mel_head(latent), top_k, do_sample)  # (1, 1)
        self.latents.index_copy_(0, self.mel_pos, latent)
        self.out_tokens.index_copy_(0, self.mel_pos, nxt[0])
        self.seen.index_fill_(0, nxt[0], True)
        self.tok.copy_(nxt)
        self.pos.add_(1)
        self.mel_pos.add_(1)

    def _graph(self, top_k, do_sample):
        key = (top_k, do_sample)
        if key in self.graphs:
            self.graphs[key] = self.graphs.pop(key)  # mark most recently used
        else:
            if len(self.graphs) >= MAX_GRAPHS:
                self.graphs.pop(next(iter(self.graphs)))
            # Warm up on a side stream (required before capture), then record. State is reset by each prefill.
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            saved = [t.clone() for t in (self.tok, self.pos, self.mel_pos, self.seen)]
            with torch.cuda.stream(s):
                for _ in range(2):
                    self._step(top_k, do_sample)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool, capture_error_mode="thread_local"):
                self._step(top_k, do_sample)
            self.pool = g.pool()
            for t, v in zip((self.tok, self.pos, self.mel_pos, self.seen), saved):
                t.copy_(v)
            self.graphs[key] = g
        return self.graphs[key]

    # ---------- generation ----------

    def _prefill(self, prefix_emb, top_k, do_sample, temperature, top_p, repetition_penalty):
        """Run conditioning + text + start token through the model; sample audio token 0's successor."""
        g = self.gpt
        start = torch.full((1, 1), g.start_audio_token, dtype=torch.long, device=self.device)
        x = torch.cat([prefix_emb.to(self.dtype), g.mel_embedding(start) + g.mel_pos_embedding.emb.weight[:1][None]], dim=1)
        T = x.shape[1]
        if T + self.max_new >= self.max_len:
            raise ValueError(f"prompt too long: {T} tokens")
        positions = self.arange_len[:T]
        mask = self.arange_len[None, :] <= positions[:, None]
        self.temperature.fill_(temperature)
        self.top_p.fill_(top_p)
        self.rep_penalty.fill_(repetition_penalty)
        # HF applies the repetition penalty to every id in input_ids: the prefix placeholder (1) and the start token
        self.seen.zero_()
        self.seen[1] = True
        self.seen[g.start_audio_token] = True
        latent = self._forward(x, positions, mask)[:, -1]
        nxt = self._sample(g.mel_head(latent), top_k, do_sample)
        self.latents[0] = latent[0]
        self.out_tokens[0] = nxt[0, 0]
        self.seen[nxt[0, 0]] = True
        self.tok.copy_(nxt)
        self.pos.fill_(T)
        self.mel_pos.fill_(1)

    def run(self, prefix_emb, top_k=50, top_p=0.85, temperature=0.75, repetition_penalty=5.0, do_sample=True):
        """Yields (tokens, latents) in blocks: tokens (n,), latents (n, C). The last block ends with the stop token."""
        top_k = int(top_k or 0)
        graph = self._graph(top_k, do_sample)  # capture before prefill: capture warmup touches the state
        self._prefill(prefix_emb, top_k, do_sample, temperature, top_p, repetition_penalty)
        done = 0  # tokens already yielded
        n = 1  # tokens sampled so far
        while True:
            # Every token sampled so far is in out_tokens[:n]; check for stop, emit, then generate more.
            toks = self.out_tokens[done:n]
            hits = (toks == self.stop).nonzero()
            if hits.numel():
                end = done + int(hits[0, 0]) + 1
                yield self.out_tokens[done:end].clone(), self.latents[done:end].clone()
                return
            if n >= self.max_new:
                yield self.out_tokens[done:n].clone(), self.latents[done:n].clone()
                return
            if n > done:
                yield self.out_tokens[done:n].clone(), self.latents[done:n].clone()
                done = n
            for _ in range(min(STOP_CHECK_EVERY, self.max_new - n)):
                graph.replay()
                n += 1


def install(xtts_model):
    """Patch an XTTS model so inference() and inference_stream() use the CUDA-graph decoder."""
    gpt = xtts_model.gpt
    fast = FastGPT(gpt)

    def _kwargs(kw):
        if kw.get("num_beams", 1) != 1:
            raise ValueError("fast decoder supports num_beams=1 only")
        return dict(
            top_k=kw.get("top_k", 50),
            top_p=kw.get("top_p", 1.0),
            temperature=kw.get("temperature", 1.0),
            repetition_penalty=kw.get("repetition_penalty", 1.0),
            do_sample=kw.get("do_sample", True),
        )

    def generate(cond_latents, text_inputs, **kw):
        gpt.compute_embeddings(cond_latents, text_inputs)  # stores prefix embeddings on gpt_inference
        toks = [t for t, _ in fast.run(gpt.gpt_inference.cached_prefix_emb, **_kwargs(kw))]
        return torch.cat(toks)[None]

    def get_generator(fake_inputs, **kw):
        # compute_embeddings already ran (inference_stream calls it first); match HF stream: one (token, latent) at a time
        for toks, lats in fast.run(gpt.gpt_inference.cached_prefix_emb, **_kwargs(kw)):
            for i in range(toks.shape[0]):
                yield toks[i : i + 1], lats[i : i + 1]

    gpt.generate = generate
    gpt.get_generator = get_generator
    xtts_model.fast_gpt = fast
    return fast
