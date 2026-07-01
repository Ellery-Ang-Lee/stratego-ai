import random
from pathlib import Path
import probability_engine
import stratego_env
import torch
import torch.nn as nn
import model
import numpy as np
import math
from torch.utils.tensorboard import SummaryWriter


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

writer = SummaryWriter("runs/rl-8")

if any(Path("models").iterdir()):
    newist = max([f for f in Path("models").iterdir() if f  .is_file()], key=lambda f: f.stat().st_mtime)
    print(str(newist).split("/")[1][0])
    print("loading model: " + str(newist))
    net.load_state_dict(torch.load(newist, map_location=device))

def main():
    global global_step, drawn_games, game_length

    for epoch in range(10000):
        print("---------- Starting game " + str(epoch) + " ----------")
        trajectory, winner = play_game()
        train_on_trajectory(trajectory, compute_returns(trajectory, winner, 0.0))
        global_step += 1

        if global_step % 400 == 0:
            torch.save(net.state_dict(), "models/rl-" + str(global_step)) 

            benchmark_wins = 0
            for i in range(20):
                benchmark_wins += validation_game()
            winrate = (1 - (benchmark_wins / 20)) * 100
            if winrate > 80:
                benchmark.load_state_dict(net.state_dict()) 
            
            writer.add_scalar("Benchmark/winrate", winrate, global_step)
            writer.add_scalar("Training/draws", drawn_games, global_step)
            writer.add_scalar("Training/game_length", game_length / 400, global_step)

            drawn_games = 0
            game_length = 0


def get_setup():
    with open("setups/red.txt", "r", encoding="utf-8") as file:
        lines = file.readlines()
        red = lines[random.randint(0,len(lines) - 1)].strip()
    
    with open("setups/blue.txt", "r", encoding="utf-8") as file:
        lines = file.readlines()
        blue = lines[random.randint(0,len(lines) - 1)].strip()
    
    return red,blue
 

def play_game():
    global drawn_games, game_length

    env = stratego_env.StrategoEnv()
    probability_engine.initialize()
    env.reset(*get_setup())

    trajectory = []
    winner = None

    done = False   
    current_turn = 0 

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

            obs = env.step(env.action_to_gravon(action))
            probability_engine.step(obs)

            turn['reward'] = 0.0 #win loss signal comes later
            turn['unknown_combat_reward'] = 0.0
            #turn['no_capture'] = obs['no_capture_count']
            if current_turn == 0:
                turn['home_distance_reward'] = (obs['home_distance_score']['red_home_distance'] / 180) - (obs['home_distance_score']['blue_home_distance'] / 180)
            else:
                turn['home_distance_reward'] = (obs['home_distance_score']['blue_home_distance'] / 180) - (obs['home_distance_score']['red_home_distance'] / 180)

            if obs['combat_outcome'] is None:
                pass
            elif obs['combat_outcome'] == 'attacker_wins':
                if not obs['newly_revealed_to']: # defender was already revealed
                    capture_value = rank_reward(obs['to_piece']) / 1.5
                else: # defender was hidden
                    capture_value = rank_reward(obs['to_piece'])
                    if stratego_env.cell_rank(obs['from_piece']) != stratego_env.SCOUT:
                        turn['unknown_combat_reward'] = 0.2 + (capture_value / 6)
                
                if obs['newly_revealed_from']: # attacker got revealed in the process
                    info_penalty = rank_reward(obs['from_piece']) / 3
                else:
                    info_penalty = 0.0
                
                turn['reward'] += capture_value - info_penalty

            elif obs['combat_outcome'] == 'defender_wins':
                if not obs['newly_revealed_from']: # attacker was already revealed
                    loss_value = rank_reward(obs['from_piece']) / 1.5
                else: # attacker was hidden
                    loss_value = rank_reward(obs['from_piece'])
                
                if obs['newly_revealed_to']: # defender got revealed in the process
                    info_penalty = rank_reward(obs['to_piece']) / 3
                    if stratego_env.cell_rank(obs['from_piece']) != stratego_env.SCOUT:
                        turn['unknown_combat_reward'] = 0.2 + (loss_value / 6)
                else:
                    info_penalty = 0.0
                
                turn['reward'] -= (loss_value - info_penalty)
            else:
                pass #ranks are always the same so its a draw

            if obs['done']:
                done = True
                winner = obs['winner']
                if winner is None:
                    drawn_games += 1
                game_length += obs['move_count']

            #turn.append(obs['reward'])

            current_turn = 1 - current_turn
            trajectory.append(turn)
    

    return trajectory, winner

def rank_reward(cell):
    rank = stratego_env.cell_rank(cell)
    return [0.00,1.00,0.30,0.10,0.30,0.20,0.30,0.40,0.45,0.50,0.60,0.70,0.30][rank]

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
        terminal = [-0.7, -0.7]
    else:
        terminal = [-1.0, -1.0]
        terminal[winner] = 1.0

    future = [0.0, 0.0]  
    for t in reversed(range(T)):
        player = t % 2
        step = trajectory[t]
        combat = step['reward'] + step['unknown_combat_reward']
        future[player] = combat + gamma * future[player]
        returns[t] = future[player] + step['home_distance_reward'] + terminal[player] - baseline

    print(returns)
    return returns
    

def train_on_trajectory(trajectory, returns):
    inputs = torch.tensor(
        np.stack([step["input"] for step in trajectory], axis=0),
        dtype=torch.float32,
        device=device,
    )

    actions = torch.tensor(
        [step["action"] for step in trajectory],
        dtype=torch.long,
        device=device,
    )

    returns = torch.tensor(
        returns,
        dtype=torch.float32,
        device=device,
    )
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    logits = net(inputs)[0]

    distribution = torch.distributions.Categorical(logits=logits)
    log_probs = distribution.log_prob(actions)

    entropy = distribution.entropy().mean()
    total_loss = -(log_probs * returns).mean() - 0.02 * entropy

    optim.zero_grad()
    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

    optim.step()


main()