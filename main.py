import websockets 
from websockets.server import serve, broadcast
import numpy as np
from pathlib import Path
import probability_engine
import asyncio
import json
import stratego_env
import torch
import time
import model
import math

#type turn data blue

env = stratego_env.StrategoEnv()

net = model.Net().to("cpu")
net.eval()

if any(Path("models").iterdir()):
    newist = max([f for f in Path("models").iterdir() if f.is_file()], key=lambda f: f.stat().st_mtime)
    print(str(newist).split("/")[1][0])
    print("loading model: " + str(newist))
    net.load_state_dict(torch.load(newist, map_location="cpu"))

connections = set()

prev_red_distance = None
prev_blue_distance = None

async def process(websocket):
    global prev_red_distance, prev_blue_distance
    connections.add(websocket)
    async for data in websocket:
        print(f"Received: {data}")
        message = json.loads(data)

        if message["type"] == "game":
            setup = message["data"]
                    
            env.reset(red_setup=setup[:60], blue_setup=setup[60:])
            probability_engine.initialize()

            print("game reset")
        
        elif message["type"] == "move":
            try:
                from_str = stratego_env.rc_to_gravon(*stratego_env.pos_to_rc(int(message["data"].split(",")[0])))
                to_str = stratego_env.rc_to_gravon(*stratego_env.pos_to_rc(int(message["data"].split(",")[1])))
                print(from_str)
                print(to_str)
                obs = env.step(from_str + "-" + to_str)
                probability_engine.step(obs)

                state = env.generate_website_board()                

                broadcast(connections, json.dumps(generate_payload("move", message["data"])))

                if obs['combat_outcome'] is not None:
                    time.sleep(2)

                broadcast(connections, json.dumps(generate_payload("game", state)))

                if obs['winner'] is not None:
                    if obs['winner'] == 0:
                        broadcast(connections, json.dumps(generate_payload("win", "red")))
                    else:
                        broadcast(connections, json.dumps(generate_payload("win", "blue")))


            except ValueError:
                broadcast(connections, json.dumps(generate_payload("move", "invalid"))) 
        elif message["type"] == "turn":
            if message["data"] == "blue":
                time.sleep(1)
                with torch.no_grad():
                    pred = net(
                        torch.tensor(
                            np.expand_dims(probability_engine.generate_input(1), axis=0), #1 is blue
                            dtype=torch.float32,
                            device="cpu"
                        )
                    )
                
                mask_tensor = torch.tensor(env.legal_actions_mask(), device="cpu")
                additive_mask = torch.where(mask_tensor, 0.0, float('-inf'))

                temperature = 0.4
                masked_logits = (pred[0] + additive_mask) / temperature

                probabilities = torch.softmax(masked_logits, dim=1)
                action = torch.multinomial(probabilities, num_samples=1).item()
                obs = env.step(env.action_to_gravon(action))
                probability_engine.step(obs)

                if obs["combat_outcome"] is not None:
                    broadcast(connections, json.dumps(generate_payload("reveal", stratego_env.rc_to_pos(*obs["from_rc"]))))

                    time.sleep(2)


                state = env.generate_website_board()

                broadcast(connections, json.dumps(generate_payload("game", state)))

                if obs['winner'] is not None:
                    if obs['winner'] == 0:
                        broadcast(connections, json.dumps(generate_payload("win", "red")))
                    else:
                        broadcast(connections, json.dumps(generate_payload("win", "blue")))
                else:
                    broadcast(connections, json.dumps(generate_payload("turn", "red")))

def generate_payload(type, data):
    return {
        "type" : type,
        "data" : data
    }

async def main():
    async with serve(process, "0.0.0.0", 8081):
        print("WebSocket server running on ws://localhost:8765")
        await asyncio.Future()  

if __name__ == "__main__":
    asyncio.run(main())