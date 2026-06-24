import os
import xml.etree.ElementTree as ET
from pathlib import Path

for folder in os.listdir("training_data"):
    if not Path("training_data/" + folder).is_dir():
        continue
    for file in os.listdir(os.path.join("training_data", folder)):
        print(file)
        if not str(file).split("-")[0] == "classic":
                continue
        root = ET.parse('training_data/' + str(folder) + '/' + str(file)).getroot()

        for game in root.findall('game'):
            if game.find('field') is None: 
                continue

            if not (game.find('result').get('type') == "1" or game.find('result').get('type') == "3"):
                continue

            winner = int(game.find('result').get('winner')) - 1
            setup = game.find('field').get("content")[::-1]
            temp = ""
            for i in range(10):
                temp += setup[(i * 10):10 + (i * 10)][::-1]
            setup = temp

            if winner == 0:
                with open("setups/red.txt", "a", encoding="utf-8") as file:
                    file.write(setup[:60] + "\n")
            else:
                with open("setups/blue.txt", "a", encoding="utf-8") as file:
                    file.write(setup[60:] + "\n")
            
