import stratego_env
import probability_engine
import model
import xml.etree.ElementTree as ET
import os
import numpy as np 

#REMEMBER: SHUFFLE MULTIPLE GAMES INTO BATCH SO ITS LESS NOISY

def main():
    env = stratego_env.StrategoEnv()
    net = model.Net()

    batch_size = 32
    current_batch_size = 0
    current_batch = np.zeros((32, 8, 26, 10, 10))
    current_value_labels = np.zeros((32, 1))
    current_policy_labels = np.zeros((32, 1))

    for folder in os.listdir("training_data"):
        for file in os.listdir(os.path.join("training_data", folder)):
            if not str(file).split("-")[0] == "classic":
                continue

            print("\n\n")
            print("reading from file: " + str(file))

            root = ET.parse('training_data/' + str(folder) + '/' + str(file)).getroot()
    
            for game in root.findall('game'):

                if not (game.find('result').get('type') == "1" or game.find('result').get('type') == "3"):
                    continue
                
                winner = int(game.find('result').get('winner')) - 1 #0 means red won, 1 means blue won

                setup = game.find('field').get("content")[::-1]
                temp = ""
                for i in range(10):
                    temp += setup[(i * 10):10 + (i * 10)][::-1]
                setup = temp
            
                env.reset(red_setup=setup[:60], blue_setup=setup[60:])
                probability_engine.initialize()

                #env.render(True)
        
                for move in game.findall('move'):
                    obs = env.step(move.get("source") + "-" + move.get("target"))
                    probability_engine.step(obs)
                    current_batch[current_batch_size] = probability_engine.generate_input(obs['current_player'])
                    current_value_labels[current_batch_size] = 1 if obs['current_player'] == winner else -1
                    #current_policy_labels[current_batch_size] = <some way to convert from gravon to 0-9999
                    
                    current_batch_size += 1
                    if current_batch_size == batch_size:
                        current_batch_size = 0
                        #give to model, calculate loss, etc...
    return


main()