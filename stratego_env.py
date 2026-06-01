#board encoding: 1 - 12 and -1 - -12. 0 is empty and 99 is lake. 
#flag = 1, bomb = 12
#colum a-j (left - right)
#row 1-10 (row 1 = red home at bottom, row 10 = blue home at top)
#action encoding: 
#  int: from_pos * 100 + to_pos, where pos = row*10 + col (range 0-9999)
#  tuple: (from_row, from_col, to_row, to_col)
#  str: gravon move string, e.g. "d3-d4"

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


def encode_piece(player: int, rank: int) -> int:
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

def rc_to_pos(r: int, c: int) -> int:
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

def rc_to_gravon(r: int, c: int) -> str:
    #row, col to gravon string
    return f"{_COLS[c]}{10 - r}"




