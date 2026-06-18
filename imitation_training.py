import stratego_env
import probability_engine
import model
import xml.etree.ElementTree as ET
import os
from pathlib import Path
import numpy as np 
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

version = "4"
game_count = 1500
global_step = 29833
batch_size = 16
validation = True

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("On device " + str(device))

env = stratego_env.StrategoEnv()
net = model.Net().to(device)

net.train()

CEL = nn.CrossEntropyLoss()
MSE = nn.MSELoss()

optim = torch.optim.Adam(
    net.parameters(),
    lr = 0.004
)

def main():
    global game_count, global_step, batch_size
    if validation:
        for checkpoint in Path("models").iterdir():
            net.load_state_dict(torch.load(checkpoint, map_location=device))
            print(str(checkpoint) + ": " + str(validate(None)))
        return
    else:
        writer = SummaryWriter("runs/experiment_" + version)

        if any(Path("models").iterdir()):
            newist = max([f for f in Path("models").iterdir() if f.is_file()], key=lambda f: f.stat().st_mtime)
            print(str(newist).split("/")[1][0])
            if str(newist).split("/")[1][0] == version:
                print("loading model: " + str(newist))
                net.load_state_dict(torch.load(newist, map_location=device))

        current_batch_size = 0
        current_batch = np.zeros((32, 8, 26, 10, 10))
        #current_value_labels = np.zeros((32, 1))
        current_policy_labels = np.zeros((32,), dtype=np.int64)
        
        for epoch in range(20):
            for folder in os.listdir("training_data"):
                for file in os.listdir(os.path.join("training_data", folder)):
                    if not str(file).split("-")[0] == "classic":
                        continue

                    print("\nreading from file: " + str(file))

                    root = ET.parse('training_data/' + str(folder) + '/' + str(file)).getroot()
            
                    for game in root.findall('game'):
                        print("----- game: " + str(game_count) + " -----")

                        if not (game.find('result').get('type') == "1" or game.find('result').get('type') == "3"):
                            continue

                        if game.find('field') is None:
                            continue
                        
                        winner = int(game.find('result').get('winner')) - 1 #0 means red won, 1 means blue won

                        setup = game.find('field').get("content")[::-1]
                        temp = ""
                        for i in range(10):
                            temp += setup[(i * 10):10 + (i * 10)][::-1]
                        setup = temp
                    
                        env.reset(red_setup=setup[:60], blue_setup=setup[60:])
                        probability_engine.initialize()
                
                        for move,next_move in zip(game.findall('move'), game.findall('move')[1:]):
                            obs = env.step(move.get("source") + "-" + move.get("target"))
                            probability_engine.step(obs)
                            current_batch[current_batch_size] = probability_engine.generate_input(obs['current_player'])
                            #current_value_labels[current_batch_size] = 1 if obs['current_player'] == winner else -1
                            current_policy_labels[current_batch_size] = stratego_env.gravon_to_policy(next_move.get("source"),next_move.get("target"))
                            
                            current_batch_size += 1
                            if current_batch_size == batch_size:
                                current_batch_size = 0
                                
                                pred = net(
                                    torch.tensor(
                                        current_batch,
                                        dtype=torch.float32,
                                        device=device
                                    )
                                )

                                policy_loss = CEL(pred[0],torch.LongTensor(current_policy_labels).to(device))
                                #value_loss = MSE(pred[1], torch.FloatTensor(current_value_labels).to(device))
                                #total_loss = (policy_loss / 9.2) #+ value_loss #5 is a constant to weight the value and policy
                                total_loss = policy_loss

                                optim.zero_grad()

                                total_loss.backward()

                                optim.step()

                                global_step += 1
                                writer.add_scalar("Loss/policy", policy_loss.item(), global_step)
                                #writer.add_scalar("Loss/value", value_loss.item(), global_step)

                        
                        game_count += 1
                        
                        if game_count % 250 == 0:
                            validate(writer)

                        if game_count % 500 == 0 and game_count != 0:
                            torch.save(net.state_dict(), "models/" + version + "-" + str(game_count)) 
        return

def validate(writer = None):
    net.eval()

    total_loss = 0
    move_count = 0
    current_batch_size = 0

    current_batch = np.zeros((32, 8, 26, 10, 10))
    current_policy_labels = np.zeros((32,), dtype=np.int64)

    for folder in os.listdir("validation_data"):
        for file in os.listdir(os.path.join("validation_data", folder)):
            if not str(file).split("-")[0] == "classic":
                continue
            root = ET.parse('validation_data/' + str(folder) + '/' + str(file)).getroot()
            for game in root.findall('game'):
                if not (game.find('result').get('type') == "1" or game.find('result').get('type') == "3"):
                    continue
                if game.find('field') is None: continue
                setup = game.find('field').get("content")[::-1]
                temp = ""
                for i in range(10):
                    temp += setup[(i * 10):10 + (i * 10)][::-1]
                setup = temp
        
                env.reset(red_setup=setup[:60], blue_setup=setup[60:])
                probability_engine.initialize()

                for move,next_move in zip(game.findall('move'), game.findall('move')[1:]):
                    obs = env.step(move.get("source") + "-" + move.get("target"))
                    probability_engine.step(obs)
                    print(env.generate_website_board())
                    current_batch[current_batch_size] = probability_engine.generate_input(obs['current_player'])
                    current_policy_labels[current_batch_size] = stratego_env.gravon_to_policy(next_move.get("source"),next_move.get("target"))

                    current_batch_size += 1

                    if current_batch_size == batch_size:
                        current_batch_size = 0

                        with torch.no_grad():
                            pred = net(
                                torch.tensor(
                                    current_batch,
                                    dtype=torch.float32,
                                    device=device
                                )
                            )

                        target = torch.tensor(
                            current_policy_labels,
                            dtype=torch.long,
                            device=device
                        )

                        policy_loss = CEL(pred[0], target)

                        total_loss += policy_loss.item()
                        move_count += 1
    
    if writer is not None:
        writer.add_scalar("Loss/validation", total_loss / move_count, global_step)
    else:
        return total_loss / move_count

    net.train()

main()