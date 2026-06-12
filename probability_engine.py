import numpy as np
import stratego_env

DEAD = 0
BELIEF_LEN = 13 #[dead, FLAG, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, bomb]
H = 8  #history

piece_beliefs = {}  #each piece id has a array of beliefs

# piece_ids[H-1] is the most recent board
piece_ids = [[-1] * 100 for _ in range(H)]

_INITIAL_COUNTS = {rank: count for rank, count in stratego_env.PIECE_COUNTS.items()}
red_piece_counts  = dict(_INITIAL_COUNTS)
blue_piece_counts = dict(_INITIAL_COUNTS)

def unknown_prior(counts):
    vec = np.zeros(BELIEF_LEN, dtype=float)
    total = sum(counts.values())
    for rank, count in counts.items():
        vec[rank] = count / total
    return vec

def renormalize(vec):
    s = vec.sum()
    if s == 0:
        return dead_encoding()
    return vec / s

def dead_encoding():
    return np.zeros(BELIEF_LEN, dtype=float)

def one_hot(rank):
    vec = np.zeros(BELIEF_LEN, dtype=float)
    vec[rank] = 1.0
    return vec


def initialize():
    global piece_beliefs, piece_ids, red_piece_counts, blue_piece_counts

    piece_beliefs = {}
    piece_ids = [[-1] * 100 for _ in range(H)]
    red_piece_counts  = dict(_INITIAL_COUNTS)
    blue_piece_counts = dict(_INITIAL_COUNTS)


def counts_for(player):
    return blue_piece_counts if player == stratego_env.RED else red_piece_counts #flipped because gravon


def _propagate_eliminations(player):
    counts = counts_for(player)

    for pid, vec in piece_beliefs.items():
        if vec.sum() == 0:
            continue
        if _piece_player.get(pid) != player:
            continue
        if vec.max() == 1.0:
            continue

        new_vec = vec.copy()
        total = sum(list(counts.values())) 
        for rank, count in counts.items():
            new_vec[rank] = count/total 

        piece_beliefs[pid] = renormalize(new_vec)

_piece_player = {}

def _register_piece(piece):
    rank = stratego_env.cell_rank(piece)
    player = stratego_env.cell_player(piece)
    if player is None:
        return

    pid = piece.id
    if pid in piece_beliefs:
        return
    
    _piece_player[pid] = player
    counts = counts_for(player)

    if piece.revealed:
        piece_beliefs[pid] = one_hot(rank)
    else:
        piece_beliefs[pid] = unknown_prior(counts)


def step(obs):
    global piece_ids

    from_piece = obs['from_piece']
    to_piece = obs['to_piece']
    combat_outcome = obs['combat_outcome']

    _register_piece(obs['from_piece'])

    # Register new stuff
    board = obs['board']
    for r in range(10):
        for c in range(10):
            _register_piece(board[r, c])

    from_pid = from_piece.id
    from_player = stratego_env.cell_player(from_piece)

    #move = not flag or bomb
    if obs['newly_moved_from']:
        vec = piece_beliefs[from_pid].copy()
        vec[stratego_env.FLAG] = 0.0
        vec[stratego_env.BOMB] = 0.0
        piece_beliefs[from_pid] = renormalize(vec)

    #move_distance > 1 = scout
    if obs['move_distance'] > 1:
        piece_beliefs[from_pid] = one_hot(stratego_env.SCOUT)

    # combat reveals
    if combat_outcome is not None:
        from_rank = stratego_env.cell_rank(from_piece)
        from_player = stratego_env.cell_player(from_piece)

        if obs['newly_revealed_from']:
            piece_beliefs[from_pid] = one_hot(from_rank)
            counts = counts_for(from_player)
            counts[from_rank] -= 1

        if to_piece is not None:
            to_pid = to_piece.id
            to_rank = stratego_env.cell_rank(to_piece)
            to_player = stratego_env.cell_player(to_piece)

            if obs['newly_revealed_to']:
                piece_beliefs[to_pid] = one_hot(to_rank)
                counts = counts_for(to_player)
                counts[to_rank] -= 1

        # deal with the dead
        if combat_outcome == 'attacker_wins':
            piece_beliefs[to_piece.id] = dead_encoding()
        elif combat_outcome == 'defender_wins':
            piece_beliefs[from_pid] = dead_encoding()
        else: 
            piece_beliefs[from_pid] = dead_encoding()
            piece_beliefs[to_piece.id] = dead_encoding()

    _propagate_eliminations(stratego_env.RED)
    _propagate_eliminations(stratego_env.BLUE)

    snapshot = [-1] * 100
    for r in range(10):
        for c in range(10):
            cell = board[r, c]
            player = stratego_env.cell_player(cell)
            if player is not None:
                snapshot[r * 10 + c] = cell.id

    piece_ids.pop(0)
    piece_ids.append(snapshot)


def generate_input():
    result = []
    K = len(next(iter(piece_beliefs.values())))
    for h in piece_ids:
        lookup = np.zeros((max(piece_beliefs) + 1, K))

        for key, value in piece_beliefs.items():
            if key == -1:
                continue
            lookup[key] = value

        ids = np.array(h).reshape(10, 10)

        result.append(lookup[ids].transpose(2,0,1))
    
    result = np.stack(result)
    
    print(np.shape(result)) #(8, 10, 10, 13)
    print(result)

    return result
    
    
    # H * 25 * 10 * 10