#board reps: 1 - 12 and -1 - -12. 0 is empty and 99 is lake. 
#flag = 1, bomb = 12
#colum a-j (left - right)
#row 1-10 (row 1 = red home at bottom, row 10 = blue home at top)

#probability engine:
# - moved? Bomb, Flag
# - moved more than one? Scout
# - change probabilities based on number of pieces remaining? or AI can do that, not sure

import numpy as np
import math

FLAG, SPY, SCOUT, MINER, SERGEANT = 1, 2, 3, 4, 5
LIEUTENANT, CAPTAIN, MAJOR, COLONEL = 6, 7, 8, 9
GENERAL, MARSHAL, BOMB = 10, 11, 12

EMPTY = 0
LAKE  = 99

GRAVON_TO_RANK = {
    "B" : BOMB, "N" : BOMB, "C" : SPY, "O" : SPY,
    "D" : SCOUT, "P" : SCOUT, "E" : MINER, "Q" : MINER,
    "F" : SERGEANT, "R" : SERGEANT, "G" : LIEUTENANT, "S" : LIEUTENANT,
    "H" : CAPTAIN, "T" : CAPTAIN, "I" : MAJOR, "U" : MAJOR,
    "J" : COLONEL, "V" : COLONEL, "K" : GENERAL, "W" : GENERAL,
    "L" : MARSHAL, "X" : MARSHAL, "M" : FLAG, "Y" : FLAG,
    "A" : EMPTY, "_" : LAKE
}

RANK_SYMBOL = {
    FLAG:'F', SPY:'1', SCOUT:'2', MINER:'3', SERGEANT:'4',
    LIEUTENANT:'5', CAPTAIN:'6', MAJOR:'7', COLONEL:'8',
    GENERAL:'9', MARSHAL:'M', BOMB:'B', 
}

RED  = 0   # Gravon rows 1-4 (0-indexed rows 6-9)
BLUE = 1   # Gravon rows 7-10 (0-indexed rows 0-3)

PIECE_COUNTS = {
    FLAG:1, SPY:1, SCOUT:8, MINER:5, SERGEANT:4,
    LIEUTENANT:4, CAPTAIN:4, MAJOR:3, COLONEL:2,
    GENERAL:1, MARSHAL:1, BOMB:6,
}

DRAW_MOVE_LIMIT = 400

_COLS = 'ABCDEFGHIK' #skipping j bc gravon is bad



def encode_piece(player, rank):
    #convert from player and rank to cell values
    return rank if player == RED else -rank

def cell_player(cell):
    #returns player based on cell
    if 1 <= cell <= 12:   return RED
    if -12 <= cell <= -1: return BLUE
    return None

def cell_rank(cell):
    #returns rank based on cell
    if 1 <= cell <= 12:   return cell 
    if -12 <= cell <= -1: return -cell
    return None

def rc_to_pos(r, c):
    #(row, col) to linear position 0-99
    return r * 10 + c

def pos_to_rc(pos):
    #linear position to row, col
    return pos // 10, pos % 10

def gravon_to_rc(s):
    #gavon string like d3 to row, col
    col = _COLS.index(s[0].upper())
    if s[1:].isdigit():
        row = 10 - int(s[1:])
    else:
        row = 0
    return row, col

def rc_to_gravon(r, c):
    #row, col to gravon string
    if r == 0:
        return f"{_COLS[c]}:"
    return f"{_COLS[c]}{10 - r}"


class StrategoEnv:
    def reset(self, red_setup: str = None, blue_setup: str = None):
        #setup mapping (row, col) to rank in a dict. None for random

        self.board = np.zeros((10, 10), dtype=np.int32)
        self.revealed = np.zeros((10, 10), dtype=bool)
        self.moved = np.zeros((10, 10), dtype=bool)
        self.current_player = RED
        self.done = False
        self.winner = None       
        self.move_count = 0
        self.no_capture_count = 0

        if red_setup is None: red_setup = self._random_setup(RED)
        if blue_setup is None: blue_setup = self._random_setup(BLUE)
            
        i = 0
        for char in red_setup:
            self.board[math.floor(i / 10),i % 10] = encode_piece(BLUE, GRAVON_TO_RANK[char])
            i += 1
        i = 0
        for char in blue_setup:
            self.board[math.floor(i / 10) + 6, i % 10] = encode_piece(RED, GRAVON_TO_RANK[char])
            i += 1

        return self._observe()

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

        self.moved[fr, fc] = True

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
                self.board[fr, fc] = EMPTY
                self.revealed[fr, fc] = False
                self.board[tr, tc] = moving_cell   # attacker advances
                self.revealed[tr, tc] = True
                self.moved[tr, tc] = True
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

    def legal_actions(self):
        #legal actions as integers 
        return sorted(self._gen_legal_actions())

    def legal_actions_mask(self):
        #every possible move, true when legal
        mask = np.zeros(10_000, dtype=bool)
        for a in self._gen_legal_actions():
            mask[a] = True
        return mask

    def action_to_gravon(self, action):
        #int to string
        fr, fc, tr, tc = self._parse_action(action)
        return f"{rc_to_gravon(fr,fc)}-{rc_to_gravon(tr,tc)}"

    def gravon_to_action(self, move_str):
        #string to int
        a, b = move_str.lower().split('-')
        fr, fc = gravon_to_rc(a)
        tr, tc = gravon_to_rc(b)
        return rc_to_pos(fr, fc) * 100 + rc_to_pos(tr, tc)

    def render(self, reveal_all = True):
        R, B, RST = '\033[91m', '\033[94m', '\033[0m'
        print("\n    a  b  c  d  e  f  g  h  i  j")
        for r in range(10):
            print(f"{10-r:2d}  ", end='')
            for c in range(10):
                cell = self.board[r, c]
                if cell == LAKE or cell == -LAKE:
                    print('~~ ', end='')
                elif cell == EMPTY:
                    print(' . ', end='')
                else:
                    player = cell_player(cell)
                    rank = cell_rank(cell)
                    #sym = RANK_SYMBOL[rank.item()] if (reveal_all or self.revealed[r, c]) else '?'
                    sym = "T" if self.moved[r, c] else "F"
                    color = R if player == RED else B
                    print(f"{color}{sym:>2}{RST} ", end='')
            print()
        player_str = 'RED' if self.current_player == RED else 'BLUE'
        print(f"\n  Turn: {player_str}   Move #{self.move_count}")



    def _observe(self) -> dict:
        return {
            'board': self.board.copy(),
            'revealed': self.revealed.copy(),
            'current_player': self.current_player,
            'legal_actions_mask': self.legal_actions_mask(),
        }

    def _move_piece(self, fr: int, fc: int, tr: int, tc: int):
        self.board[tr, tc]  = self.board[fr, fc]
        self.revealed[tr, tc] = self.revealed[fr, fc]
        self.moved[tr, tc] = self.moved[fr, fc]
        self.board[fr, fc]  = EMPTY
        self.revealed[fr, fc] = False

    def _resolve_combat(self, atk: int, dfn: int) -> str:
        if dfn == BOMB:
            return 'attacker_wins' if atk == MINER else 'defender_wins'
        if dfn == FLAG:
            return 'attacker_wins'
        if atk == SPY and dfn == MARSHAL:
            return 'attacker_wins'
        if atk > dfn: return 'attacker_wins'
        if atk < dfn: return 'defender_wins'
        return 'draw'

    def _gen_legal_actions(self) -> set:
        actions = set()
        dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))

        for r in range(10):
            for c in range(10):
                cell = self.board[r, c]
                if cell_player(cell) != self.current_player:
                    continue

                rank = cell_rank(cell)
                from_pos = rc_to_pos(r, c)

                if rank in (FLAG, BOMB):
                    continue   # immovable pieces

                if rank == SCOUT:
                    for dr, dc in dirs:
                        nr, nc = r + dr, c + dc
                        while 0 <= nr < 10 and 0 <= nc < 10:
                            target = self.board[nr, nc]
                            if target == LAKE:
                                break
                            to_pos = rc_to_pos(nr, nc)
                            if target == EMPTY:
                                actions.add(from_pos * 100 + to_pos)
                                nr += dr
                                nc += dc
                            else:
                                if cell_player(target) != self.current_player:
                                    actions.add(from_pos * 100 + to_pos)
                                break 
                else:
                    for dr, dc in dirs:
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < 10 and 0 <= nc < 10):
                            continue
                        target = self.board[nr, nc]
                        if target == LAKE:
                            continue
                        if target == EMPTY or cell_player(target) != self.current_player:
                            actions.add(from_pos * 100 + rc_to_pos(nr, nc))

        return actions

    def _has_legal_moves(self) -> bool:
        #game ends in a tie?
        dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))
        for r in range(10):
            for c in range(10):
                cell = self.board[r, c]
                if cell_player(cell) != self.current_player:
                    continue
                rank = cell_rank(cell)
                if rank in (FLAG, BOMB):
                    continue
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < 10 and 0 <= nc < 10):
                        continue
                    target = self.board[nr, nc]
                    if target != LAKE and (target == EMPTY or
                                           cell_player(target) != self.current_player):
                        return True
        return False

    def _is_legal(self, fr: int, fc: int, tr: int, tc: int) -> bool:
        return (rc_to_pos(fr, fc) * 100 + rc_to_pos(tr, tc)) in self._gen_legal_actions()

    def _parse_action(self, action):
        if isinstance(action, str):
            a, b   = action.lower().split('-')
            fr, fc = gravon_to_rc(a)
            tr, tc = gravon_to_rc(b)
        elif isinstance(action, int):
            from_pos, to_pos = action // 100, action % 100
            fr, fc = pos_to_rc(from_pos)
            tr, tc = pos_to_rc(to_pos)
        elif isinstance(action, (tuple, list)):
            fr, fc, tr, tc = action
        else:
            raise ValueError(f"Unsupported action type: {type(action)}")
        return fr, fc, tr, tc

    def _random_setup(self, player: int) -> dict:
        rows    = range(6, 10) if player == RED else range(0, 4)
        squares = [(r, c) for r in rows for c in range(10)]
        pieces  = [rank for rank, cnt in PIECE_COUNTS.items()
                   for _ in range(cnt)]
        np.random.shuffle(pieces)
        return dict(zip(squares, pieces))