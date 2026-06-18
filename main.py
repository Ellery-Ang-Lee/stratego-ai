import websockets 
from websockets.server import serve
import asyncio
import json

#type: game, move

async def process(client):
    async for message in client:
        print(f"Received: {message}")


async def main():
    async with serve(process, "172.16.43.101", 8081):
        print("WebSocket server running on ws://localhost:8765")
        await asyncio.Future()  

if __name__ == "__main__":
    asyncio.run(main())