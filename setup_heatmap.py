import math

RANK_TO_GR = {
    "B" : 11,
    "C" : 1,
    "D" : 2,
    "E" : 3,
    "F" : 4,
    "G" : 5,
    "H" : 6,
    "I" : 7,
    "J" : 8,
    "K" : 9,
    "L" : 10, 
    "M" : 0,
}

RANK_TO_NAME = ['flag', 'spy', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'bomb']
heatmap = []

for i in range(10):
    heatmap.append([[0 for k in range(12)] for j in range(10)])


with open("setups/blue.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()
    for line in lines:
        i = 0
        for char in line.strip():
            heatmap[math.floor(i / 10)][i % 10][RANK_TO_GR[char]] += 1
            i += 1


for i in range(12):
    print("\n\nPiece: " + RANK_TO_NAME[i])
    result = []
    for j in range(10):
        for k in range(10):
            result.append(heatmap[j][k][i])

    #flipped = []
    #for j in range(100):
    #    flipped.append(result[99-j])
    
    flipped = result[:40]

    max_val = max(flipped)
    
    for j in range(40):
        color = '\033[38;2;' + str(math.floor(flipped[j] * 255 / max_val))  +';' + str(math.floor(0.20 * (flipped[j] * 255 / max_val))) + ';0m'
        # print(color + str(flipped[j]), end=" " * (6 - len(str(flipped[j]))))
        print(color + '██', end="")
        if (j + 1) % 10 == 0 and j != 0:
            print("")

print('\033[0m')
            
        

import torch
sd = torch.load("models/rl-800")  # or whatever the latest checkpoint before the crash was
for name, param in sd.items():
    if torch.isnan(param).any() or torch.isinf(param).any():
        print(f"NaN/Inf found in {name}")
