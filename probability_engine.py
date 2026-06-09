#import torch
import stratego_env

#H * (10 * 10) * 27 input size
# 12 channels for blue pieces, 12 channels for red pieces, 1 channel for lakes, 1 channel for empty, 1 channel for known or unknown 

#initialize piece_ids to 10 * 10 * 10 with every entry being -1
piece_ids = [[[-1 for _ in range(10)] for _ in range(10)] for _ in range(8)] #8 * 10 * H for the IDs

piece_belifs = {}

def initialize():
    global piece_ids, piece_belifs

    piece_ids = [[[-1 for _ in range(10)] for _ in range(10)] for _ in range(8)]
    for i in range(100):
        if i in [40,41,50,51,44,45,54,55,48,49,58,59]:
            piece_belifs[i] = [0,0,0,0,0,0,0,0,0,0,0,0,0,1]
        elif i in [42,43,52,53,46,47,56,57]:
            piece_belifs[i] = [0,0,0,0,0,0,0,0,0,0,0,0,1,0]
        else:
            piece_belifs[i] = [1/30,1/30,8/30,5/30,4/30,4/30,4/30,3/30,2/30,1/30,1/30,0,0]
                            #flat, 1-10, bomb, lake, empty
    
def step(obs):
    piece_ids.pop(0)
    temp = []
    for i in range(10):
        for j in range(10):
            temp.append(obs["board"][i][j].id)
    piece_ids.append(temp)

    print(piece_ids)

    


def generate_input():
    pass