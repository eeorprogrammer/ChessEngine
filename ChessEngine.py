#!/usr/bin/env python3
"""
engine.py
A compact, readable UCI chess engine in Python.

- Move generation & legality: python-chess
- Search: Iterative Deepening + Negamax + Alpha-Beta + Transposition Table
- Eval: Material + PST + mobility + simple pawn structure
- Time management: wtime/btime/movestogo/movetime
- UCI protocol: uci / isready / ucinewgame / position / go / stop / quit

Install deps:
    pip install python-chess

Run (console):
    python engine.py

Use with a UCI GUI (e.g., CuteChess):
    Add engine: path/to/python path/to/engine.py

Notes:
- Minimal comments focus on *why*.
- Written for clarity & maintainability over raw speed.
"""
from __future__ import annotations

import sys
import time
import threading
import math
import dataclasses
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import chess  # type: ignore


# -----------------------------
# Search Parameters & Defaults
# -----------------------------
INF = 10_000_000
MATE_SCORE = 9_000_000
MATE_IN_MAX = 10000

DEFAULT_MOVE_OVERHEAD_MS = 50  # buffer to avoid time forfeits
MAX_PLY = 128


# -----------------------------
# Transposition Table
# -----------------------------
class Bound:
    EXACT = 0
    LOWER = 1  # fail-high
    UPPER = 2  # fail-low


@dataclass
class TTEntry:
    key: int
    depth: int
    value: int
    bound: int
    best_move: Optional[chess.Move]


class TranspositionTable:
    """Fixed-size TT with replacement by depth."""

    def __init__(self, size_mb: int = 64) -> None:
        entries = (size_mb * 1024 * 1024) // dataclasses.asdict(TTEntry(0, 0, 0, 0, None)).__sizeof__()
        # Use a power-of-two size for cheap indexing.
        self.mask = 1
        while self.mask < max(1024, entries):
            self.mask <<= 1
        self.mask -= 1
        self.table: List[Optional[TTEntry]] = [None] * (self.mask + 1)

    def _index(self, key: int) -> int:
        return key & self.mask

    def probe(self, key: int) -> Optional[TTEntry]:
        e = self.table[self._index(key)]
        return e if e and e.key == key else None

    def store(self, entry: TTEntry) -> None:
        i = self._index(entry.key)
        cur = self.table[i]
        # Prefer higher depth; avoids thrashing shallow info.
        if cur is None or entry.depth >= cur.depth:
            self.table[i] = entry


# -----------------------------
# Evaluation
# -----------------------------
# PST values (middlegame) adapted for readability. Tune later.
# Units are in centipawns.
PST_PAWN = [
      0,  0,  0,  0,  0,  0,  0,  0,
     50, 50, 50, 50, 50, 50, 50, 50,
     10, 10, 20, 30, 30, 20, 10, 10,
      5,  5, 10, 25, 25, 10,  5,  5,
      0,  0,  0, 20, 20,  0,  0,  0,
      5, -5,-10,  0,  0,-10, -5,  5,
      5, 10, 10,-20,-20, 10, 10,  5,
      0,  0,  0,  0,  0,  0,  0,  0,
]

PST_KNIGHT = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]

PST_BISHOP = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]

PST_ROOK = [
      0,  0,  5, 10, 10,  5,  0,  0,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
      5, 10, 10, 10, 10, 10, 10,  5,
      0,  0,  0,  0,  0,  0,  0,  0,
]

PST_QUEEN = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  5,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]

PST_KING_MG = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]

PST_KING_EG = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
]

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

PIECE_PSTS = {
    chess.PAWN: PST_PAWN,
    chess.KNIGHT: PST_KNIGHT,
    chess.BISHOP: PST_BISHOP,
    chess.ROOK: PST_ROOK,
    chess.QUEEN: PST_QUEEN,
}


def game_phase(board: chess.Board) -> float:
    """Return phase in [0..1], 0=opening/mid, 1=endgame. Used for king PST blend.
    We use remaining non-pawn material as a proxy.
    """
    total = 4 * PIECE_VALUES[chess.KNIGHT] + 4 * PIECE_VALUES[chess.BISHOP] + 4 * PIECE_VALUES[chess.ROOK] + 2 * PIECE_VALUES[chess.QUEEN]
    white = (
        len(board.pieces(chess.KNIGHT, True)) * PIECE_VALUES[chess.KNIGHT]
        + len(board.pieces(chess.BISHOP, True)) * PIECE_VALUES[chess.BISHOP]
        + len(board.pieces(chess.ROOK, True)) * PIECE_VALUES[chess.ROOK]
        + len(board.pieces(chess.QUEEN, True)) * PIECE_VALUES[chess.QUEEN]
    )
    black = (
        len(board.pieces(chess.KNIGHT, False)) * PIECE_VALUES[chess.KNIGHT]
        + len(board.pieces(chess.BISHOP, False)) * PIECE_VALUES[chess.BISHOP]
        + len(board.pieces(chess.ROOK, False)) * PIECE_VALUES[chess.ROOK]
        + len(board.pieces(chess.QUEEN, False)) * PIECE_VALUES[chess.QUEEN]
    )
    remaining = white + black
    return max(0.0, min(1.0, 1.0 - remaining / max(1, total)))


def eval_board(board: chess.Board) -> int:
    """Static evaluation from the side to move perspective.
    Uses simple, fast terms to be search-friendly.
    """
    if board.is_checkmate():
        return -MATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    phase = game_phase(board)

    score = 0

    # Material + PSTs.
    for piece_type in PIECE_VALUES:
        if piece_type == chess.KING:
            continue
        w_bb = board.pieces(piece_type, True)
        b_bb = board.pieces(piece_type, False)
        score += chess.popcount(w_bb) * PIECE_VALUES[piece_type]
        score -= chess.popcount(b_bb) * PIECE_VALUES[piece_type]
        pst = PIECE_PSTS[piece_type]
        for sq in w_bb:
            score += pst[sq]
        for sq in b_bb:
            score -= pst[chess.square_mirror(sq)]  # mirror for black

    # King safety/placement blended by phase.
    wk = board.king(True)
    bk = board.king(False)
    if wk is not None and bk is not None:
        mg = PST_KING_MG[wk] - PST_KING_MG[chess.square_mirror(bk)]
        eg = PST_KING_EG[wk] - PST_KING_EG[chess.square_mirror(bk)]
        score += int((1 - phase) * mg + phase * eg)

    # Mobility: encourages active play; helps avoid horizon pathologies.
    score += 2 * (board.legal_moves.count() if board.turn else -board.legal_moves.count())

    # Simple pawn structure: doubled/isolated penalties.
    score += pawn_structure(board)

    return score if board.turn else -score


def pawn_structure(board: chess.Board) -> int:
    score = 0
    for color in [True, False]:
        pawns = board.pieces(chess.PAWN, color)
        files = [0] * 8
        for sq in pawns:
            files[chess.square_file(sq)] += 1
        # Doubled pawns.
        doubled = sum(max(0, c - 1) for c in files)
        if doubled:
            score += (-12 * doubled) if color else (12 * doubled)
        # Isolated pawns.
        for f, c in enumerate(files):
            if c == 0:
                continue
            adj = (files[f - 1] if f - 1 >= 0 else 0) + (files[f + 1] if f + 1 < 8 else 0)
            if adj == 0:
                score += (-10 * c) if color else (10 * c)
    return score


# -----------------------------
# Move Ordering Heuristics
# -----------------------------
MVV_LVA = [[0] * 7 for _ in range(7)]
for victim in range(1, 7):
    for attacker in range(1, 7):
        MVV_LVA[victim][attacker] = 10 * victim - attacker


class MoveOrderer:
    def __init__(self) -> None:
        self.killers: List[List[Optional[chess.Move]]] = [[None, None] for _ in range(MAX_PLY)]
        self.history: Dict[Tuple[bool, int, int], int] = {}

    def score(self, board: chess.Board, move: chess.Move, tt_move: Optional[chess.Move], ply: int) -> int:
        if tt_move is not None and move == tt_move:
            return 1_000_000
        if board.is_capture(move):
            victim = board.piece_type_at(move.to_square) or 0
            attacker = board.piece_type_at(move.from_square) or 0
            return 100_000 + MVV_LVA[victim][attacker]
        # Killer moves: worked before in this ply.
        k1, k2 = self.killers[ply]
        if move == k1:
            return 80_000
        if move == k2:
            return 70_000
        # History heuristic: quiet moves that refuted branches.
        key = (board.turn, move.from_square, move.to_square)
        return self.history.get(key, 0)

    def update_killers(self, ply: int, move: chess.Move) -> None:
        k1, k2 = self.killers[ply]
        if k1 != move:
            self.killers[ply] = [move, k1]

    def update_history(self, board: chess.Board, move: chess.Move, depth: int) -> None:
        key = (board.turn, move.from_square, move.to_square)
        self.history[key] = self.history.get(key, 0) + depth * depth


# -----------------------------
# Search Engine
# -----------------------------
@dataclass
class SearchLimits:
    wtime: Optional[int] = None  # ms
    btime: Optional[int] = None  # ms
    movestogo: Optional[int] = None
    movetime: Optional[int] = None  # ms
    depth: Optional[int] = None


class StopToken:
    def __init__(self) -> None:
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def stopped(self) -> bool:
        return self._stop


class Engine:
    def __init__(self, tt_size_mb: int = 128) -> None:
        self.board = chess.Board()
        self.tt = TranspositionTable(tt_size_mb)
        self.order = MoveOrderer()
        self.nodes = 0
        self.start_time = 0.0
        self.stop_token = StopToken()
        self.best_move: Optional[chess.Move] = None
        self.seldepth = 0

    # -------------------------
    # Time Management
    # -------------------------
    def allocate_time(self, limits: SearchLimits) -> int:
        if limits.movetime is not None:
            return max(1, limits.movetime - DEFAULT_MOVE_OVERHEAD_MS)
        my_time = limits.wtime if self.board.turn else limits.btime
        if my_time is None or my_time <= 0:
            return 100  # minimal thinking
        mtg = limits.movestogo if limits.movestogo and limits.movestogo > 0 else 30
        slice_ms = max(10, my_time // mtg)
        return max(1, slice_ms - DEFAULT_MOVE_OVERHEAD_MS)

    # -------------------------
    # Search Loop
    # -------------------------
    def search(self, limits: SearchLimits) -> chess.Move:
        self.nodes = 0
        self.seldepth = 0
        self.stop_token = StopToken()
        time_budget = self.allocate_time(limits)
        self.start_time = time.monotonic()
        deadline = self.start_time + time_budget / 1000.0

        best_move = None
        best_score = -INF
        tt_entry = self.tt.probe(self.board.zobrist_hash())
        tt_move = tt_entry.best_move if tt_entry else None

        max_depth = limits.depth if limits.depth is not None else 64
        aspiration = 50  # centipawns window; narrows re-search cost

        for depth in range(1, max_depth + 1):
            alpha, beta = best_score - aspiration, best_score + aspiration
            score, move = self._iter(depth, alpha, beta, deadline)
            if self.stop_token.stopped():
                break
            # Aspiration failed => re-search full window.
            if score <= alpha or score >= beta:
                score, move = self._iter(depth, -INF, INF, deadline)
                if self.stop_token.stopped():
                    break
            if move is not None:
                best_move, best_score = move, score
                self.best_move = move
                # UCI info line for GUIs.
                print(
                    f"info depth {depth} seldepth {self.seldepth} score cp {score} nodes {self.nodes} time {int((time.monotonic()-self.start_time)*1000)} pv {self._pv(move)}",
                    flush=True,
                )
            # End if out of time.
            if time.monotonic() > deadline:
                break

        if best_move is None:
            # No legal move => mate or stalemate.
            moves = list(self.board.legal_moves)
            if not moves:
                return chess.Move.null()
            best_move = moves[0]
        return best_move

    def _iter(self, depth: int, alpha: int, beta: int, deadline: float) -> Tuple[int, Optional[chess.Move]]:
        score, move = self._negamax(depth, 0, alpha, beta, deadline)
        return score, move

    def _time_up(self, deadline: float) -> bool:
        return time.monotonic() >= deadline

    def _negamax(self, depth: int, ply: int, alpha: int, beta: int, deadline: float) -> Tuple[int, Optional[chess.Move]]:
        if self._time_up(deadline):
            self.stop_token.stop()
            return 0, None
        self.seldepth = max(self.seldepth, ply)
        self.nodes += 1

        key = self.board.zobrist_hash()
        tt_hit = self.tt.probe(key)
        if tt_hit and tt_hit.depth >= depth:
            val = tt_hit.value
            if tt_hit.bound == Bound.EXACT:
                return val, tt_hit.best_move
            if tt_hit.bound == Bound.LOWER:
                alpha = max(alpha, val)
            elif tt_hit.bound == Bound.UPPER:
                beta = min(beta, val)
            if alpha >= beta:
                return val, tt_hit.best_move

        if depth == 0:
            # Quiescence search to reduce horizon effect; capture-only.
            return self._quiescence(alpha, beta, ply, deadline), None

        if self.board.is_checkmate():
            return -MATE_SCORE + ply, None
        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0, None

        best_val = -INF
        best_move: Optional[chess.Move] = None

        # Move ordering (TT move, captures, killers, history).
        tt_move = tt_hit.best_move if tt_hit else None
        moves = list(self.board.legal_moves)
        moves.sort(key=lambda m: self.order.score(self.board, m, tt_move, ply), reverse=True)

        for move in moves:
            self.board.push(move)
            val, _ = self._negamax(depth - 1, ply + 1, -beta, -alpha, deadline)
            val = -val
            self.board.pop()

            if self.stop_token.stopped():
                return 0, None

            if val > best_val:
                best_val = val
                best_move = move
            if val > alpha:
                alpha = val
                # Quiet improvement => good killer/history signal.
                if not self.board.is_capture(move):
                    self.order.update_history(self.board, move, depth)
            if alpha >= beta:
                if not self.board.is_capture(move):
                    self.order.update_killers(ply, move)
                break

        # Store in TT with proper bound type.
        bound = Bound.EXACT if best_val > alpha and best_val < beta else (Bound.LOWER if best_val >= beta else Bound.UPPER)
        self.tt.store(TTEntry(key=key, depth=depth, value=best_val, bound=bound, best_move=best_move))
        return best_val, best_move

    def _quiescence(self, alpha: int, beta: int, ply: int, deadline: float) -> int:
        if self._time_up(deadline):
            self.stop_token.stop()
            return 0
        stand_pat = eval_board(self.board)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        # Only consider captures; ordered by MVV-LVA.
        captures = [m for m in self.board.legal_moves if self.board.is_capture(m)]
        captures.sort(key=lambda m: MVV_LVA[self.board.piece_type_at(m.to_square) or 0][self.board.piece_type_at(m.from_square) or 0], reverse=True)

        for move in captures:
            self.board.push(move)
            val = -self._quiescence(-beta, -alpha, ply + 1, deadline)
            self.board.pop()

            if self.stop_token.stopped():
                return 0

            if val >= beta:
                return beta
            if val > alpha:
                alpha = val
        return alpha

    def _pv(self, first_move: chess.Move) -> str:
        # Shallow PV trace by following TT best moves.
        pv_moves = [first_move]
        b = self.board.copy()
        b.push(first_move)
        for _ in range(10):
            e = self.tt.probe(b.zobrist_hash())
            if not e or not e.best_move or e.best_move not in b.legal_moves:
                break
            pv_moves.append(e.best_move)
            b.push(e.best_move)
        return " ".join(m.uci() for m in pv_moves)


# -----------------------------
# UCI Protocol
# -----------------------------
class UCI:
    def __init__(self) -> None:
        self.engine = Engine()
        self.thread: Optional[threading.Thread] = None

    def loop(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            cmd, *rest = line.split()

            if cmd == "uci":
                print("id name PyNegamaxTT", flush=True)
                print("id author Code Copilot", flush=True)
                print("option name Hash type spin default 128 min 16 max 1024", flush=True)
                print("uciok", flush=True)

            elif cmd == "isready":
                print("readyok", flush=True)

            elif cmd == "setoption":
                # Only Hash supported for simplicity.
                try:
                    name_idx = rest.index("name") + 1
                    value_idx = rest.index("value") + 1
                    name = rest[name_idx]
                    value = int(rest[value_idx])
                    if name.lower() == "hash":
                        self.engine = Engine(tt_size_mb=value)
                except Exception:
                    pass

            elif cmd == "ucinewgame":
                self.engine = Engine()

            elif cmd == "position":
                self._handle_position(rest)

            elif cmd == "go":
                limits = self._parse_go(rest)
                self._start_search(limits)

            elif cmd == "stop":
                self._stop_search()

            elif cmd == "quit":
                self._stop_search()
                return

    def _handle_position(self, tokens: List[str]) -> None:
        try:
            if tokens[0] == "startpos":
                self.engine.board = chess.Board()
                moves_idx = 1
            elif tokens[0] == "fen":
                fen = " ".join(tokens[1:7])
                self.engine.board = chess.Board(fen)
                moves_idx = 7
            else:
                return
            if moves_idx < len(tokens) and tokens[moves_idx] == "moves":
                for u in tokens[moves_idx + 1 : ]:
                    self.engine.board.push_uci(u)
        except Exception:
            # Ignore malformed positions to keep engine responsive.
            pass

    def _parse_go(self, tokens: List[str]) -> SearchLimits:
        it = iter(tokens)
        limits = SearchLimits()
        for t in it:
            if t == "wtime":
                limits.wtime = int(next(it))
            elif t == "btime":
                limits.btime = int(next(it))
            elif t == "movestogo":
                limits.movestogo = int(next(it))
            elif t == "movetime":
                limits.movetime = int(next(it))
            elif t == "depth":
                limits.depth = int(next(it))
        return limits

    def _start_search(self, limits: SearchLimits) -> None:
        self._stop_search()
        def worker() -> None:
            move = self.engine.search(limits)
            if move is None:
                print("bestmove 0000", flush=True)
            else:
                print(f"bestmove {move.uci()}", flush=True)
        self.thread = threading.Thread(target=worker, daemon=True)
        self.thread.start()

    def _stop_search(self) -> None:
        if self.engine and not self.engine.stop_token.stopped():
            self.engine.stop_token.stop()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.05)


if __name__ == "__main__":
    UCI().loop()
