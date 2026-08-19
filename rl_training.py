import random
from pathlib import Path
import probability_engine
import stratego_env
import torch
import torch.nn as nn
import model
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import Pool, get_context
import io

global_step = 0 #UNCOMMENT THE CHECKPOINT RESTORE 

# UNCOMMENT THE CHECKPOINT RESTORE 
# UNCOMMENT THE CHECKPOINT RESTORE 
# UNCOMMENT THE CHECKPOINT RESTORE 

device = None
net = None
benchmark = None
optim = None
writer = None

drawn_games = 0
game_length = 0
entropy_sum = 0.0
entropy_count = 0

running_mean = 0.0
running_var = 1.0
running_initialized = False
norm_momentum = 0.005 

def main():
    global global_step, drawn_games, game_length, entropy_sum, entropy_count
    global device, net, benchmark, optim, writer

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("RL on " + str(device))

    net = model.Net().to(device)
    net.train()

    benchmark = model.Net().to(device)
    benchmark.load_state_dict(net.state_dict())
    benchmark.eval()

    optim = torch.optim.Adam(
        net.parameters(),
        lr = 0.0003
    )

    writer = SummaryWriter("runs/rl-30")

    #if any(Path("models").iterdir()):
    #    newist = max([f for f in Path("models").iterdir() if f.is_file()], key=lambda f: f.stat().st_mtime)
    #    print("loading model: " + str(newist))
    #    net.load_state_dict(torch.load(newist, map_location=device))

    ctx = get_context("spawn")

    with ctx.Pool(processes=10) as pool:
        for epoch in range(99999):
            print("---------- Starting round " + str(epoch) + " ----------")

            torch.save(net.state_dict(), "temp_learner.pt")

            checkpoint_pool = [f for f in Path("models").iterdir() if f.is_file()]

            batch_args = []
            results = []

            print("Selecting opponents:")

            for _ in range(50):
                if checkpoint_pool and random.random() < 0.5:
                    opponent_path = str(random.choice(checkpoint_pool))
                    print(" - " + opponent_path)
                else:
                    opponent_path = None
                    print(" - Self Play")
                batch_args.append(("temp_learner.pt", opponent_path))

            for i in range(5):
                results += pool.map(worker_play_game, batch_args[i*10:(i+1)*10])

            round_trajectories = []
            round_returns = []
            round_meta = []

            for trajectory, winner, is_draw, move_count, learner_side in results:
                print(f"trajectory length: {len(trajectory)}")   

                returns = compute_returns(trajectory, winner, 0.0)

                learner_trajectory = [t for t in trajectory if t['is_learner']]
                learner_returns = [r for t, r in zip(trajectory, returns) if t['is_learner']]

                print(f"winner={winner}, learner_side={learner_side}, is_draw={is_draw}, mean_learner_return={np.mean(learner_returns):.3f}")

                round_trajectories.append(learner_trajectory)
                round_returns.append(learner_returns)
                round_meta.append((is_draw, move_count))

            mean_entropy = train_on_round(round_trajectories, round_returns)
            entropy_sum += mean_entropy
            entropy_count += 1

            for is_draw, move_count in round_meta:
                if is_draw:
                    drawn_games += 1
                game_length += move_count
                global_step += 1

                if global_step % 50 == 0:
                    writer.add_scalar("Training/draws", drawn_games, global_step)
                    writer.add_scalar("Training/game_length", game_length / 50, global_step)
                    writer.add_scalar("Training/entropy", entropy_sum / max(entropy_count, 1), global_step)
                    drawn_games = 0
                    game_length = 0
                    entropy_sum = 0.0
                    entropy_count = 0

                if global_step % 400 == 0:
                    torch.save(net.state_dict(), "models/rl-" + str(global_step)) 

                    benchmark_wins = 0
                    for i in range(20):
                        benchmark_wins += validation_game()
                    winrate = (1 - (benchmark_wins / 20)) * 100
                    if winrate > 70:
                        benchmark.load_state_dict(net.state_dict()) 
                    
                    writer.add_scalar("Benchmark/winrate", winrate, global_step)

            if device.type == "mps":
                torch.mps.empty_cache()


def get_setup():
    with open("setups/red.txt", "r", encoding="utf-8") as file:
        lines = file.readlines()
        red = lines[random.randint(0,len(lines) - 1)].strip()
    
    with open("setups/blue.txt", "r", encoding="utf-8") as file:
        lines = file.readlines()
        blue = lines[random.randint(0,len(lines) - 1)].strip()
    
    return red,blue
 
def worker_play_game(args):
    learner_path, opponent_path = args
    import io
    import torch
    import model
    import probability_engine
    import stratego_env

    torch.set_num_threads(1)

    learner_net = model.Net()
    learner_net.load_state_dict(torch.load(learner_path, map_location="cpu"))
    learner_net.eval()

    if opponent_path is None:
        opponent_net = learner_net
    else:
        opponent_net = model.Net()
        opponent_net.load_state_dict(torch.load(opponent_path, map_location="cpu"))
        opponent_net.eval()
    
    learner_side = random.randint(0, 1)

    return play_game(learner_net, opponent_net, learner_side, torch.device("cpu"))

def play_game(learner_net, opponent_net, learner_side, device):
    env = stratego_env.StrategoEnv()
    probability_engine.initialize()
    env.reset(*get_setup())

    trajectory = []
    winner = None
    game_length = 0

    done = False   
    current_turn = 0 

    initial_scores = env.home_distance_score()
    red_high = past_red = initial_scores['red_home_distance']
    blue_high = past_blue = initial_scores['blue_home_distance']


    with torch.inference_mode():
        while not done:

            turn = {}
            active_net = learner_net if current_turn == learner_side else opponent_net

            input = probability_engine.generate_input(current_turn)

            pred = active_net(
                torch.tensor(
                    np.expand_dims(input, axis=0), #1 is blue
                    dtype=torch.float32,
                    device=device
                )
            )

            mask_tensor = torch.tensor(env.legal_actions_mask(), device=device)

            if not mask_tensor.any():
                print(f"WARNING: no legal actions available at turn {current_turn}, move {game_length}")
                print(f"obs done: {obs.get('done') if 'obs' in dir() else 'N/A'}")
                # Force a random legal-ish fallback or treat as terminal — adjust based on what you learn
                done = True
                winner = 1 - current_turn  # or however you want to handle this edge case
                break

            additive_mask = torch.where(mask_tensor, 0.0, float('-inf'))
            temperature = 1.0
            masked_logits = (pred[0] + additive_mask) / temperature
            probabilities = torch.softmax(masked_logits, dim=1)
            action = torch.multinomial(probabilities, num_samples=1).item()

            turn['input'] = input
            turn['action'] = action
            turn['mask'] = env.legal_actions_mask()  
            turn['is_learner'] = (current_turn == learner_side)
        
            try:
                obs = env.step(env.action_to_gravon(action))
                probability_engine.step(obs)
            except ValueError:
                while True:
                    try:
                        action = random.randint(0,10000)
                        turn['action'] = action
                        obs = env.step(env.action_to_gravon(action))
                        probability_engine.step(obs)
                        break
                    except ValueError:
                        pass

            turn['reward'] = 0.0 #win loss signal comes later
            turn['unknown_combat_reward'] = 0.0
            turn['stall_penalty'] = 0.0
            turn['home_distance_reward'] = 0.0
            #turn['no_capture'] = obs['no_capture_count']

            total_count = obs['home_distance_score']['total_count']

            if total_count > 0:
                red_score = (obs['home_distance_score']['red_home_distance'] / total_count) * 2
                blue_score = (obs['home_distance_score']['blue_home_distance'] / total_count) * 2
            else:
                red_score = past_red
                blue_score = past_blue

            if current_turn == 0:
                if red_score > red_high:
                    red_high = red_score    
                    turn['home_distance_reward'] += 0.06
                if red_score > past_red:
                    turn['home_distance_reward'] += (-0.000757 * obs['home_distance_score']['total_count']) + 0.11
                past_red = red_score 

            else:
                if blue_score > blue_high:
                    blue_high = blue_score
                    turn['home_distance_reward'] += 0.06
                if blue_score > past_blue:
                    turn['home_distance_reward'] += (-0.000757 * obs['home_distance_score']['total_count']) + 0.11
                past_blue = blue_score

            
            if obs['combat_outcome'] is None:
                stall_penalty = -((obs['no_capture_count'] / stratego_env.DRAW_MOVE_LIMIT) ** 2.5) * 32
                turn['stall_penalty'] += stall_penalty
                turn['reward'] -= 0.08
            else:
                #in home distance reward so it does not bounce back to the other player 
                base = (-0.0033 * obs['home_distance_score']['total_count']) + 0.35
                if obs['combat_outcome'] == 'attacker_wins':
                    turn['home_distance_reward'] += base
                elif obs['combat_outcome'] == 'draw':
                    turn['home_distance_reward'] += base * 0.5
                else: 
                    turn['home_distance_reward'] += base * 0.1

                if obs['combat_outcome'] == 'attacker_wins':
                    turn['reward'] += rank_reward(obs['to_piece'])
                elif obs['combat_outcome'] == 'defender_wins':
                    turn['reward'] -= rank_reward(obs['from_piece'])
                else:
                    pass

            if obs['done'] or game_length >= 3000:
                done = True
                winner = obs['winner']
                game_length = obs['move_count']

            #turn.append(obs['reward'])

            current_turn = 1 - current_turn
            trajectory.append(turn)
    

    return trajectory, winner, winner is None, game_length, learner_side

def rank_reward(cell):
    rank = stratego_env.cell_rank(cell)
    return [0.00, 10.00, 0.30, 0.10, 0.45, 0.30, 0.45, 0.60, 0.70, 1.00, 3.00, 5.00, 0.50][rank]

def validation_game():
    with torch.no_grad():
        env = stratego_env.StrategoEnv()
        probability_engine.initialize()
        env.reset(*get_setup())

        done = False   
        current_turn = 0 
        net.eval()

        winner = None

        while not done:
            input = probability_engine.generate_input(current_turn)
            if current_turn == 0:
                pred = net(
                    torch.tensor(
                        np.expand_dims(input, axis=0),
                        dtype=torch.float32,
                        device=device
                    )
                )
            else:
                pred = benchmark(
                    torch.tensor(
                        np.expand_dims(input, axis=0), 
                        dtype=torch.float32,
                        device=device
                    )
                )
        
            mask_tensor = torch.tensor(env.legal_actions_mask(), device=device)
            additive_mask = torch.where(mask_tensor, 0.0, float('-inf'))
            temperature = 0.5
            masked_logits = (pred[0] + additive_mask) / temperature
            probabilities = torch.softmax(masked_logits, dim=1)
            action = torch.multinomial(probabilities, num_samples=1).item()

            obs = env.step(env.action_to_gravon(action))
            probability_engine.step(obs)

            if obs['done'] or game_length >= 3000:
                done = True
                winner = obs['winner']
            current_turn = 1 - current_turn

        net.train()

        if winner is None:
            #maybe do something about ties?
            winner = 1

    return winner



def compute_returns(trajectory, winner, baseline, gamma=0.98):
    T = len(trajectory)
    returns = [0.0] * T

    if winner is None:
        terminal = [-10, -10]
    else:
        terminal = [-5.0, -5.0]
        terminal[winner] = 5.0

    future = [0.0, 0.0]
    pending_opponent = [0.0, 0.0]

    last_index = [None, None]
    for t in range(T):
        last_index[t % 2] = t

    for t in reversed(range(T)):
        player = t % 2
        opponent = 1 - player
        step = trajectory[t]
        combat = step["reward"] + step["unknown_combat_reward"]

        total_combat = combat + pending_opponent[player]
        step_reward = total_combat + step["home_distance_reward"] + step["stall_penalty"]

        if t == last_index[player]:
            step_reward += terminal[player]

        future[player] = step_reward + gamma * future[player]
        returns[t] = future[player] - baseline
        pending_opponent[opponent] = -step["reward"]

    return returns


def train_on_round(round_trajectories, round_returns, batch_size=256):
    all_returns_flat = np.concatenate([np.array(r, dtype=np.float32) for r in round_returns if len(r) > 0])
    global_mean, global_std = update_running_stats(all_returns_flat)

    total_learner_steps = len(all_returns_flat)

    optim.zero_grad()

    entropy_total = 0.0
    entropy_batches = 0

    for trajectory, returns in zip(round_trajectories, round_returns):
        if len(trajectory) == 0:
            continue

        all_inputs = np.stack([step["input"] for step in trajectory], axis=0)
        all_actions = np.array([step["action"] for step in trajectory], dtype=np.int64)
        all_masks = np.stack([step["mask"] for step in trajectory], axis=0)
        all_returns = np.array(returns, dtype=np.float32)

        all_returns = (all_returns - global_mean) / (global_std + 1e-8)

        T = len(trajectory)

        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)

            inputs = torch.tensor(all_inputs[start:end], dtype=torch.float32, device=device)
            actions = torch.tensor(all_actions[start:end], dtype=torch.long, device=device)
            masks = torch.tensor(all_masks[start:end], device=device)
            chunk_returns = torch.tensor(all_returns[start:end], dtype=torch.float32, device=device)

            raw_logits = net(inputs)[0]
            additive_mask = torch.where(masks, 0.0, float('-inf'))
            masked_logits = raw_logits + additive_mask

            distribution = torch.distributions.Categorical(logits=masked_logits)
            log_probs = distribution.log_prob(actions)
            entropy = distribution.entropy().mean()

            entropy_total += entropy.item()
            entropy_batches += 1

            chunk_loss = -(log_probs * chunk_returns).mean() - 0.03 * entropy
            chunk_loss = chunk_loss * (end - start) / total_learner_steps

            chunk_loss.backward()

    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    optim.step()

    return entropy_total / max(entropy_batches, 1)

def update_running_stats(returns_flat):
    global running_mean, running_var, running_initialized
    batch_mean = returns_flat.mean()
    batch_var = returns_flat.var()

    if not running_initialized:
        running_mean = batch_mean
        running_var = batch_var
        running_initialized = True
    else:
        running_mean = (1 - norm_momentum) * running_mean + norm_momentum * batch_mean
        running_var = (1 - norm_momentum) * running_var + norm_momentum * batch_var

    return running_mean, np.sqrt(running_var)

if __name__ == "__main__":
    main()