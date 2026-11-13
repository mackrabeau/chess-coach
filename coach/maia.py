"""Thin wrapper around Maia-3 (github.com/CSSLab/maia3, inference only, AGPL-3.0).

Facts checked in the maia3 code (commit 1e13597):
  * Elo is NOT bucketed: each Elo is a linear blend of two learned embeddings,
    emb(e) = (e/5000) * E_a + (1 - e/5000) * E_b, clamped to [0, 5000]. So any integer
    Elo works and predictions move smoothly with it (beyond the training range they
    drift: P(best move) starts falling again above ~2700 in our probe).
  * Input per square = 12 piece planes x `history` (8) boards + Elo embeddings for the
    side to move ("self") and the opponent. Each board is tokenized from its own side
    to move's perspective (black-to-move boards are mirrored), oldest first, current
    last; missing history is padded by repeating the earliest board. The released UCI
    engine defaults to NO history (current board repeated 8x); --use-uci-history adds it.
  * The policy is over 4352 from-to(+promotion) moves in the side-to-move frame, so
    moves for black are mirrored before lookup.
"""
from __future__ import annotations

import chess
import numpy as np


class Maia:
    def __init__(self, model: str = "23m", device: str | None = None, amp: bool = True):
        import torch
        from maia3.model_registry import resolve_checkpoint_path
        from maia3.uci import load_model, parse_args
        from maia3.utils import get_all_possible_moves

        self.torch = torch
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfg = parse_args(["--model", model, "--device", device])
        cfg.checkpoint_path = resolve_checkpoint_path(cfg.model_spec)
        self.cfg = cfg
        self.device = device
        self.amp = amp and device.startswith("cuda")
        self.model = load_model(cfg)
        self.moves = get_all_possible_moves()
        self.move_index = {m: i for i, m in enumerate(self.moves)}

    # ------------------------------------------------------------------ inputs

    def tokens(self, fen: str, history_fens=None):
        """(64, 12*history + 1) input; history_fens oldest first (may be None/empty)."""
        from maia3.dataset import tokenize_board

        torch = self.torch
        boards = [chess.Board(h) for h in (history_fens or [])][-(self.cfg.history - 1):]
        boards.append(chess.Board(fen))
        t = torch.cat([tokenize_board(b) for b in boards], dim=1)
        if len(boards) < self.cfg.history:
            t = torch.cat([tokenize_board(boards[0]).repeat(1, self.cfg.history - len(boards)), t], dim=1)
        return torch.cat([t, torch.zeros(64, 1)], dim=1)  # clk_ponder slot, sliced off by the model

    def legal_mask(self, fen: str):
        from maia3.dataset import get_legal_moves_mask

        return get_legal_moves_mask(chess.Board(fen), self.move_index)

    def index(self, fen: str, uci: str) -> int:
        from maia3.utils import mirror_move

        white = chess.Board(fen).turn == chess.WHITE
        return self.move_index[uci if white else mirror_move(uci)]

    # ------------------------------------------------------------------ inference

    def policy(self, tokens, masks, self_elo, oppo_elo):
        """Move probabilities (B, 4352) for a batch; elos are ints or (B,) arrays."""
        torch = self.torch
        B = tokens.shape[0]
        se = torch.as_tensor(np.broadcast_to(self_elo, (B,)).copy(), dtype=torch.long, device=self.device)
        oe = torch.as_tensor(np.broadcast_to(oppo_elo, (B,)).copy(), dtype=torch.long, device=self.device)
        with torch.no_grad(), torch.autocast("cuda", enabled=self.amp):
            logits, _, _ = self.model(tokens.to(self.device), se, oe)
        logits = logits.float().masked_fill(~masks.to(self.device), float("-inf"))
        return torch.softmax(logits, -1)

    def embed(self, tokens, elo: int = 1900, layer: int | None = None):
        """Hidden state (B, dim_vit) at one fixed Elo for both sides, so it describes the position, not the
        rating. layer None: the final square-mean + LayerNorm vector (input to the value head); int k: the
        output of transformer block k, averaged over the 64 squares."""
        torch = self.torch
        mod = self.model.last_ln if layer is None else self.model.transformer.layers[layer]
        got = {}
        hook = mod.register_forward_hook(lambda m, i, o: got.__setitem__("x", o))
        try:
            e = torch.full((tokens.shape[0],), int(elo), dtype=torch.long, device=self.device)
            with torch.no_grad(), torch.autocast("cuda", enabled=self.amp):
                self.model(tokens.to(self.device), e, e)
        finally:
            hook.remove()
        x = got["x"].float()
        return (x if layer is None else x.mean(1)).cpu().numpy()


def difficulty(maia, fens, histories, best_moves, elos, batch: int = 256):
    """Maia difficulty per position at the mover's own Elo (both Elo inputs = elos[i]):
    p_best = P(engine best move), entropy = entropy of the move distribution (nats). Returns two arrays."""
    torch = maia.torch
    pb, ent = np.full(len(fens), np.nan), np.full(len(fens), np.nan)
    for b0 in range(0, len(fens), batch):
        idx = range(b0, min(b0 + batch, len(fens)))
        T = torch.stack([maia.tokens(fens[i], [] if histories[i] is None else list(histories[i])[-7:]) for i in idx])
        M = torch.stack([maia.legal_mask(fens[i]) for i in idx])
        e = np.clip(np.asarray([elos[i] for i in idx], float), 0, 5000).round()
        P = maia.policy(T, M, e, e).cpu().numpy()
        for j, i in enumerate(idx):
            p = P[j]
            q = p[p > 0]
            ent[i] = float(-(q * np.log(q)).sum())
            try:
                pb[i] = float(p[maia.index(fens[i], best_moves[i])])
            except KeyError:
                pass
    return pb, ent


def maia_lists(maia, nodes, mass, kmax, batch=128):
    """nodes: list of (fen_after, history(oldest first), self_elo, oppo_elo, must_include).
    Returns per node: ([(move, P)] listed, {move: P} for all moves with P >= 0.001)."""
    import chess
    import numpy as np

    torch = maia.torch
    out = []
    for b0 in range(0, len(nodes), batch):
        sub = nodes[b0:b0 + batch]
        live = [k for k, nd in enumerate(sub) if not chess.Board(nd[0]).is_game_over()]
        res = [([], {}) for _ in sub]
        if live:
            T = torch.stack([maia.tokens(sub[k][0], sub[k][1][-7:]) for k in live])
            M = torch.stack([maia.legal_mask(sub[k][0]) for k in live])
            P = maia.policy(T, M, np.array([sub[k][2] for k in live], float),
                            np.array([sub[k][3] for k in live], float)).cpu().numpy()
            for j, k in enumerate(live):
                fa = sub[k][0]
                probs = sorted(((float(P[j, maia.index(fa, m.uci())]), m.uci())
                                for m in chess.Board(fa).legal_moves), reverse=True)
                full = {m: p for p, m in probs}
                chosen, tot = [], 0.0
                for p, m in probs:
                    if tot >= mass or len(chosen) >= kmax:
                        break
                    chosen.append((m, p))
                    tot += p
                for extra in sub[k][4]:
                    if extra and extra not in {m for m, _ in chosen} and extra in full:
                        chosen.append((extra, full[extra]))
                res[k] = (chosen, {m: p for m, p in full.items() if p >= 0.001})
        out += res
    return out
