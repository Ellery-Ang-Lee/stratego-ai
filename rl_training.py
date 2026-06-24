import random
from pathlib import Path
import probability_engine
import stratego_env
import torch
import torch.nn as nn
import model
import numpy as np
import math

global_step = 0

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("RL on  " + str(device))

net = model.Net().to(device)
net.train()

optim = torch.optim.Adam(
    net.parameters(),
    lr = 0.002
)

if any(Path("models").iterdir()):
    newist = max([f for f in Path("models").iterdir() if f  .is_file()], key=lambda f: f.stat().st_mtime)
    print(str(newist).split("/")[1][0])
    print("loading model: " + str(newist))
    net.load_state_dict(torch.load(newist, map_location=device))

def main():
    pass

def get_setup():
    with open("setups/red.txt", "a", encoding="utf-8") as file:
        lines = file.readlines()
        red = lines[random.randint(0,len(lines))]
    
    with open("setups/red.txt", "a", encoding="utf-8") as file:
        lines = file.readlines()
        blue = lines[random.randint(0,len(lines))]
    
    return red,blue
 

def play_game():
    env = stratego_env.StrategoEnv()
    probability_engine.initialize()
    env.reset(*get_setup)

    trajectory = []
    winner = None

    done = False   
    current_turn = 0 

    while not done:

        turn = []

        input = probability_engine.generate_input(current_turn)

        pred = net(
            torch.tensor(
                np.expand_dims(input, axis=0), #1 is blue
                dtype=torch.float32,
                device=device
            )
        )

        mask_tensor = torch.tensor(env.legal_actions_mask(), device="cpu")
        additive_mask = torch.where(mask_tensor, 0.0, float('-inf'))
        temperature = 1.0
        masked_logits = (pred[0] + additive_mask) / temperature
        probabilities = torch.softmax(masked_logits, dim=1)
        action = torch.multinomial(probabilities, num_samples=1).item()

        turn.append(input)
        turn.append(probabilities[action]) #prob of the action

        obs = env.step(env.action_to_gravon(action))
        probability_engine.step(obs)

        if obs['done']:
            done = True
            winner = obs['winner']

        #turn.append(obs['reward'])

        current_turn = 1 - current_turn
    

    return trajectory, winner


def compute_returns(trajectory, winner, baseline):
    returns = []
    current_turn = 0
    for step in trajectory:
        if winner is None:
            raw_return = 0.0
        elif winner == current_turn:
            raw_return = 1.0
        else:
            raw_return = -1.0

        returns.append(raw_return - baseline)
    
        current_turn = 1 - current_turn
    
    return returns
    

def train_on_trajectory(trajectory, returns):
    for i in range(len(trajectory)):
        loss = -math.log(returns[i] * trajectory[i][1])
        optim.zero_grad()
        loss.backward()
        optim.step()


main()