#board encoding: 1 - 12 and -1 - -12. 0 is empty and 99 is lake. 
#flag = 1, bomb = 12
#colum a-j (left - right)
#row 1-10 (row 1 = red home at bottom, row 10 = blue home at top)

import numpy as np

FLAG, SPY, SCOUT, MINER, SERGEANT  = 0, 1, 2, 3, 4
LIEUTENANT, CAPTAIN, MAJOR, COLONEL = 5, 6, 7, 8
GENERAL, MARSHAL, BOMB              = 9, 10, 11

RANK_TO_GRAVON = {
    FLAG:'F',   SPY:'1',  SCOUT:'2',  MINER:'3',   SERGEANT:'4',
    LIEUTENANT:'5', CAPTAIN:'6', MAJOR:'7', COLONEL:'8',
    GENERAL:'9', MARSHAL:'10', BOMB:'B',
}
GRAVON_TO_RANK = {v: k for k, v in RANK_TO_GRAVON.items()}

RANK_SYMBOL = {
    FLAG:'F', SPY:'s', SCOUT:'2', MINER:'3', SERGEANT:'4',
    LIEUTENANT:'5', CAPTAIN:'6', MAJOR:'7', COLONEL:'8',
    GENERAL:'9', MARSHAL:'M', BOMB:'B',
}

EMPTY = 0
LAKE  = 99

RED  = 0   # Gravon rows 1-4   (0-indexed rows 6-9)
BLUE = 1   # Gravon rows 7-10  (0-indexed rows 0-3)

PIECE_COUNTS = {
    FLAG:1, SPY:1, SCOUT:8, MINER:5, SERGEANT:4,
    LIEUTENANT:4, CAPTAIN:4, MAJOR:3, COLONEL:2,
    GENERAL:1, MARSHAL:1, BOMB:6,
}

LAKE_SQUARES = [
    (4, 2), (4, 3), (5, 2), (5, 3),
    (4, 6), (4, 7), (5, 6), (5, 7),
]

DRAW_MOVE_LIMIT = 400

_COLS = 'abcdefghij'  


def encode_piece(player: int, rank: int):
    #convert from player and rank to cell values
    return rank + 1 if player == RED else -(rank + 1)

def cell_player(cell: int):
    #returns player based on cell
    if 1 <= cell <= 12:   return RED
    if -12 <= cell <= -1: return BLUE
    return None

def cell_rank(cell: int):
    #returns rank based on cell
    if 1 <= cell <= 12:   return cell - 1
    if -12 <= cell <= -1: return -cell - 1
    return None

def rc_to_pos(r: int, c: int):
    #(row, col) to linear position 0-99
    return r * 10 + c

def pos_to_rc(pos: int):
    #linear position to row, col
    return pos // 10, pos % 10

def gravon_to_rc(s: str):
    #gavon string like d3 to row, col
    col = _COLS.index(s[0].lower())
    row = 10 - int(s[1:])
    return row, col

def rc_to_gravon(r: int, c: int):
    #row, col to gravon string
    return f"{_COLS[c]}{10 - r}"


class StrategoEnv:
    def reset(self, red_setup: dict = None, blue_setup: dict = None):
        #setup mapping (row, col) to rank in a dict. None for random

        self.board = np.zeros((10, 10), dtype=np.int32)
        self.revealed = np.zeros((10, 10), dtype=bool)
        self.current_player = RED
        self.done = False
        self.winner = None       
        self.move_count = 0
        self.no_capture_count = 0

        if red_setup is None: red_setup = self._random_setup(RED)
        if blue_setup is None: blue_setup = self._random_setup(BLUE)

        for (r, c), rank in red_setup.items():
            self.board[r, c] = encode_piece(RED,  rank)
        for (r, c), rank in blue_setup.items():
            self.board[r, c] = encode_piece(BLUE, rank)

        return self._observe()

    def reset_from_gravon(self, red_setup: list, blue_setup: list) -> dict:
        #position as a string, rank as a string pairs

        def parse(setup):
            return {gravon_to_rc(pos): GRAVON_TO_RANK[rank]
                    for pos, rank in setup}
        return self.reset(parse(red_setup), parse(blue_setup))

    def step(self, action) -> tuple:
        #from_pos * 100 + to_pos  (pos = row*10 + col)
        #(from_row, from_col, to_row, to_col)
        #"d3-d4"

        fr, fc, tr, tc = self._parse_action(action)

        if not self._is_legal(fr, fc, tr, tc):
            raise ValueError(
                f"Illegal move: {rc_to_gravon(fr,fc)}-{rc_to_gravon(tr,tc)}"
            )

        reward = 0.0
        info   = {'combat': None}

        moving_cell = self.board[fr, fc]
        target_cell = self.board[tr, tc]
        moving_rank = cell_rank(moving_cell)

        if target_cell == EMPTY:
            self._move_piece(fr, fc, tr, tc)
            self.no_capture_count += 1

        else:
            self.no_capture_count = 0
            target_rank = cell_rank(target_cell)

            self.revealed[fr, fc] = True
            self.revealed[tr, tc] = True

            outcome = self._resolve_combat(moving_rank, target_rank)
            info['combat'] = {
                'attacker_rank': moving_rank,
                'defender_rank': target_rank,
                'outcome':       outcome,
            }

            if outcome == 'attacker_wins':
                self.board[fr, fc]    = EMPTY
                self.revealed[fr, fc] = False
                self.board[tr, tc]    = moving_cell   # attacker advances
                self.revealed[tr, tc] = True
                if target_rank == FLAG:
                    self.done   = True
                    self.winner = self.current_player
                    reward = 1.0 if self.current_player == RED else -1.0

            elif outcome == 'defender_wins':
                self.board[fr, fc]    = EMPTY         
                self.revealed[fr, fc] = False

            else: 
                self.board[fr, fc]    = EMPTY
                self.board[tr, tc]    = EMPTY
                self.revealed[fr, fc] = False
                self.revealed[tr, tc] = False

        if not self.done:
            self.current_player  = 1 - self.current_player
            self.move_count     += 1

            if not self._has_legal_moves():
                # No legal moves → current player loses
                self.done   = True
                self.winner = 1 - self.current_player
                reward = -1.0 if self.current_player == RED else 1.0

        if not self.done and self.no_capture_count >= DRAW_MOVE_LIMIT:
            self.done   = True
            self.winner = None
            reward      = 0.0

        info.update(
            move_count=self.move_count,
            current_player=self.current_player,
            winner=self.winner,
        )
        return self._observe(), reward, self.done, info
