from __future__ import annotations

from dataclasses import dataclass


FILES = "abcdefgh"
RANKS = "12345678"
STARTING_ROOK_SQUARES = {
    "a1": "Q",
    "h1": "K",
    "a8": "q",
    "h8": "k",
}


def square_to_index(square: str) -> int:
    file_idx = FILES.index(square[0])
    rank_idx = RANKS.index(square[1])
    return rank_idx * 8 + file_idx


def index_to_square(index: int) -> str:
    return f"{FILES[index % 8]}{RANKS[index // 8]}"


def square_file(index: int) -> int:
    return index % 8


def square_rank(index: int) -> int:
    return index // 8


def flip_vertical(index: int) -> int:
    return (7 - square_rank(index)) * 8 + square_file(index)


@dataclass(slots=True)
class BoardState:
    board: dict[int, str]
    side_to_move: str
    castling: str
    ep_square: int | None
    halfmove_clock: int
    fullmove_number: int

    @classmethod
    def from_fen(cls, fen: str) -> "BoardState":
        parts = fen.split()
        if len(parts) != 6:
            raise ValueError(f"Invalid FEN: {fen}")

        placement, stm, castling, ep, halfmove, fullmove = parts
        board: dict[int, str] = {}
        ranks = placement.split("/")
        if len(ranks) != 8:
            raise ValueError(f"Invalid FEN placement: {placement}")

        for rank_offset, rank_text in enumerate(reversed(ranks)):
            file_idx = 0
            for ch in rank_text:
                if ch.isdigit():
                    file_idx += int(ch)
                else:
                    board[rank_offset * 8 + file_idx] = ch
                    file_idx += 1
            if file_idx != 8:
                raise ValueError(f"Invalid FEN rank: {rank_text}")

        ep_square = None if ep == "-" else square_to_index(ep)
        return cls(
            board=board,
            side_to_move=stm,
            castling="" if castling == "-" else castling,
            ep_square=ep_square,
            halfmove_clock=int(halfmove),
            fullmove_number=int(fullmove),
        )

    def to_fen(self) -> str:
        rank_parts: list[str] = []
        for rank in range(7, -1, -1):
            empty = 0
            text = []
            for file_idx in range(8):
                square = rank * 8 + file_idx
                piece = self.board.get(square)
                if piece is None:
                    empty += 1
                    continue
                if empty:
                    text.append(str(empty))
                    empty = 0
                text.append(piece)
            if empty:
                text.append(str(empty))
            rank_parts.append("".join(text))

        castling = self.castling or "-"
        ep = "-" if self.ep_square is None else index_to_square(self.ep_square)
        return " ".join(
            [
                "/".join(rank_parts),
                self.side_to_move,
                castling,
                ep,
                str(self.halfmove_clock),
                str(self.fullmove_number),
            ]
        )

    def copy(self) -> "BoardState":
        return BoardState(
            board=dict(self.board),
            side_to_move=self.side_to_move,
            castling=self.castling,
            ep_square=self.ep_square,
            halfmove_clock=self.halfmove_clock,
            fullmove_number=self.fullmove_number,
        )

    def piece_count(self) -> int:
        return len(self.board)

    def apply_uci(self, uci: str) -> "BoardState":
        next_state = self.copy()
        from_sq = square_to_index(uci[:2])
        to_sq = square_to_index(uci[2:4])
        promotion = uci[4] if len(uci) == 5 else None

        moving_piece = next_state.board.pop(from_sq)
        is_white = moving_piece.isupper()
        moving_side = "w" if is_white else "b"
        if moving_side != self.side_to_move:
            raise ValueError(f"Move {uci} does not match side to move in {self.to_fen()}")

        captured_piece = next_state.board.get(to_sq)
        is_capture = captured_piece is not None

        if moving_piece.upper() == "K" and abs(square_file(from_sq) - square_file(to_sq)) > 1:
            if to_sq > from_sq:
                rook_from = from_sq + 3
                rook_to = from_sq + 1
            else:
                rook_from = from_sq - 4
                rook_to = from_sq - 1
            rook_piece = next_state.board.pop(rook_from)
            next_state.board[rook_to] = rook_piece

        if (
            moving_piece.upper() == "P"
            and self.ep_square is not None
            and to_sq == self.ep_square
            and captured_piece is None
            and square_file(from_sq) != square_file(to_sq)
        ):
            captured_sq = to_sq - 8 if is_white else to_sq + 8
            next_state.board.pop(captured_sq, None)
            is_capture = True

        next_state.board[to_sq] = moving_piece
        if promotion is not None:
            next_state.board[to_sq] = promotion.upper() if is_white else promotion.lower()

        next_state._update_castling_rights(from_sq, to_sq, moving_piece, captured_piece)

        if moving_piece.upper() == "P" and abs(square_rank(from_sq) - square_rank(to_sq)) == 2:
            offset = 8 if is_white else -8
            next_state.ep_square = from_sq + offset
        else:
            next_state.ep_square = None

        if moving_piece.upper() == "P" or is_capture:
            next_state.halfmove_clock = 0
        else:
            next_state.halfmove_clock += 1

        next_state.side_to_move = "b" if self.side_to_move == "w" else "w"
        if self.side_to_move == "b":
            next_state.fullmove_number += 1

        return next_state

    def _update_castling_rights(
        self,
        from_sq: int,
        to_sq: int,
        moving_piece: str,
        captured_piece: str | None,
    ) -> None:
        rights = set(self.castling)
        from_name = index_to_square(from_sq)
        to_name = index_to_square(to_sq)

        if moving_piece == "K":
            rights.discard("K")
            rights.discard("Q")
        elif moving_piece == "k":
            rights.discard("k")
            rights.discard("q")

        if moving_piece.upper() == "R" and from_name in STARTING_ROOK_SQUARES:
            rights.discard(STARTING_ROOK_SQUARES[from_name])

        if captured_piece is not None and to_name in STARTING_ROOK_SQUARES:
            rights.discard(STARTING_ROOK_SQUARES[to_name])

        order = "KQkq"
        self.castling = "".join(ch for ch in order if ch in rights)
