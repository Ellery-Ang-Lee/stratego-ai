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

global_step = 1

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
    lr = 0.002
)

drawn_games = 0
game_length = 0

writer = SummaryWriter("runs/rl-20")

if any(Path("models").iterdir()):
    newist = max([f for f in Path("models").iterdir() if f  .is_file()], key=lambda f: f.stat().st_mtime)
    print(str(newist).split("/")[1][0])
    print("loading model: " + str(newist))
    net.load_state_dict(torch.load(newist, map_location=device))

def main():
    global global_step, drawn_games, game_length

    ctx = get_context("spawn")

    with ctx.Pool(processes=8) as pool:
        for epoch in range(99999):
            print("---------- Starting round " + str(epoch) + " ----------")

            buffer = io.BytesIO()
            torch.save(net.state_dict(), buffer)
            state_bytes = buffer.getvalue()

            results = pool.map(worker_play_game, [state_bytes] * 8)

            for trajectory, winner, is_draw, move_count in results:
                print(f"trajectory length: {len(trajectory)}")   
                train_on_trajectory(trajectory, compute_returns(trajectory, winner, 0.0))

                if is_draw:
                    drawn_games += 1
                game_length += move_count

                if device.type == "mps":
                    torch.mps.empty_cache()
                global_step += 1

                if global_step % 50 == 0:
                    writer.add_scalar("Training/draws", drawn_games, global_step)
                    writer.add_scalar("Training/game_length", game_length / 50, global_step)
                    drawn_games = 0
                    game_length = 0 

                if global_step % 400 == 0:
                    torch.save(net.state_dict(), "models/rl-" + str(global_step)) 

                    benchmark_wins = 0
                    for i in range(20):
                        benchmark_wins += validation_game()
                    winrate = (1 - (benchmark_wins / 20)) * 100
                    if winrate > 80:
                        benchmark.load_state_dict(net.state_dict()) 
                    
                    writer.add_scalar("Benchmark/winrate", winrate, global_step)


def get_setup():
    with open("setups/red.txt", "r", encoding="utf-8") as file:
        lines = file.readlines()
        red = lines[random.randint(0,len(lines) - 1)].strip()
    
    with open("setups/blue.txt", "r", encoding="utf-8") as file:
        lines = file.readlines()
        blue = lines[random.randint(0,len(lines) - 1)].strip()
    
    return red,blue
 
def worker_play_game(state_dict_bytes):
    import io
    import torch
    import model
    import probability_engine
    import stratego_env

    worker_net = model.Net()
    buffer = io.BytesIO(state_dict_bytes)
    worker_net.load_state_dict(torch.load(buffer, map_location="cpu"))
    worker_net.eval()

    return play_game(worker_net, torch.device("cpu"))

def play_game(net, device):
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

            input = probability_engine.generate_input(current_turn)

            pred = net(
                torch.tensor(
                    np.expand_dims(input, axis=0), #1 is blue
                    dtype=torch.float32,
                    device=device
                )
            )

            mask_tensor = torch.tensor(env.legal_actions_mask(), device=device)
            additive_mask = torch.where(mask_tensor, 0.0, float('-inf'))
            temperature = 1.0
            masked_logits = (pred[0] + additive_mask) / temperature
            probabilities = torch.softmax(masked_logits, dim=1)
            action = torch.multinomial(probabilities, num_samples=1).item()

            turn['input'] = input
            turn['action'] = action
            
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

            red_score = (obs['home_distance_score']['red_home_distance'] / obs['home_distance_score']['total_count']) * 2
            blue_score = (obs['home_distance_score']['blue_home_distance'] / obs['home_distance_score']['total_count']) * 2

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
                stall_penalty = -((obs['no_capture_count'] / stratego_env.DRAW_MOVE_LIMIT) ** 4) * 100
                turn['stall_penalty'] += stall_penalty
            else:
                #in home distance reward so it does not bounce back to the other player 
                base = (-0.00333 * obs['home_distance_score']['total_count']) + 0.32
                if obs['combat_outcome'] == 'attacker_wins':
                    turn['home_distance_reward'] += base
                elif obs['combat_outcome'] == 'draw':
                    turn['home_distance_reward'] += base * 0.5
                else: 
                    turn['home_distance_reward'] += base * 0.1

                if obs['combat_outcome'] == 'attacker_wins':
                    if not obs['newly_revealed_to']: # defender was already revealed
                        capture_value = rank_reward(obs['to_piece']) / 1.2
                    else: # defender was hidden
                        capture_value = rank_reward(obs['to_piece'])
                        #if stratego_env.cell_rank(obs['from_piece']) != stratego_env.SCOUT:
                        turn['unknown_combat_reward'] = 0.05 - (capture_value / 6)
                    
                    if obs['newly_revealed_from']: # attacker got revealed in the process
                        info_penalty = rank_reward(obs['from_piece']) / 7
                    else:
                        info_penalty = 0.0
                    
                    turn['reward'] += capture_value - info_penalty

                elif obs['combat_outcome'] == 'defender_wins':
                    if not obs['newly_revealed_from']: # attacker was already revealed
                        loss_value = rank_reward(obs['from_piece']) / 1.2
                    else: # attacker was hidden
                        loss_value = rank_reward(obs['from_piece'])
                    
                    if obs['newly_revealed_to']: # defender got revealed in the process
                        info_penalty = rank_reward(obs['to_piece']) / 7
                    #    if stratego_env.cell_rank(obs['from_piece']) != stratego_env.SCOUT:
                        turn['unknown_combat_reward'] = 0.05 + (loss_value / 6)
                    else:
                        info_penalty = 0.0
                    
                    turn['reward'] -= (loss_value - info_penalty)
                else:
                    pass

            if obs['done']:
                done = True
                winner = obs['winner']
                game_length = obs['move_count']

            #turn.append(obs['reward'])

            current_turn = 1 - current_turn
            trajectory.append(turn)
    

    return trajectory, winner, winner is None, game_length

def rank_reward(cell):
    rank = stratego_env.cell_rank(cell)
    return [0.00,5.00,0.40,0.10,0.15,0.20,0.30,0.40,0.45,0.50,0.60,0.70,0.50][rank]

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

            if obs['done']:
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
        terminal = [-1.1, -1.1]
    else:
        terminal = [-1.0, -1.0]
        terminal[winner] = 1.0

    future = [0.0, 0.0]
    pending_opponent = [0.0, 0.0]

    for t in reversed(range(T)):
        player = t % 2
        opponent = 1 - player
        step = trajectory[t]
        combat = step["reward"] + step["unknown_combat_reward"]

        total_combat = combat + pending_opponent[player]
        future[player] = total_combat + gamma * future[player]

        returns[t] = (future[player] + step["home_distance_reward"] + step['stall_penalty'] + terminal[player] - baseline)
        pending_opponent[opponent] = -step["reward"]

    return returns
    

def train_on_trajectory(trajectory, returns, batch_size=256):
    all_inputs = np.stack([step["input"] for step in trajectory], axis=0)
    all_actions = np.array([step["action"] for step in trajectory], dtype=np.int64)
    all_returns = np.array(returns, dtype=np.float32)

    all_returns = (all_returns - all_returns.mean()) / (all_returns.std() + 1e-8)

    T = len(trajectory)
    optim.zero_grad()

    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)

        inputs = torch.tensor(all_inputs[start:end], dtype=torch.float32, device=device)
        actions = torch.tensor(all_actions[start:end], dtype=torch.long, device=device)
        chunk_returns = torch.tensor(all_returns[start:end], dtype=torch.float32, device=device)

        logits = net(inputs)[0]
        distribution = torch.distributions.Categorical(logits=logits)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy().mean()

        #scale it so one chuck is equal to the total loss of the trajectory
        chunk_loss = -(log_probs * chunk_returns).mean() - 0.06 * entropy
        chunk_loss = chunk_loss * (end - start) / T

        chunk_loss.backward()

    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    optim.step()

if __name__ == "__main__":
    main()


