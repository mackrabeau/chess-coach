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
