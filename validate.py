import numpy as np

piece_beliefs = {1: [0,1,0], 2: [0.5,0.5,0.25], 3: [1,0,1], 4: [2,5,2]}
piece_ids = [[1,3,2,4]]


def generate_input():
    result = []
    K = len(next(iter(piece_beliefs.values())))
    for h in piece_ids:
        lookup = np.zeros((max(piece_beliefs) + 1, K))

        for key, value in piece_beliefs.items():
            if key == -1:
                continue
            lookup[key] = value

        ids = np.array(h).reshape(2, 2)

        result.append(lookup[ids].transpose(2,0,1))
    
    result = np.stack(result)
    
    print(np.shape(result)) 
    print(result)

    return result

[[0,  1  ,0.5  ,2],
 [1,  0  ,0.5  ,5],
 [0,  1  ,0.25 ,2]]


generate_input()