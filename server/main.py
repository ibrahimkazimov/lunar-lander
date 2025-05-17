import asyncio
import websockets
import json
import random
# from rl_agent import RLAgent

# agent = RLAgent()

async def handle_connection(websocket):
    print("Client connected!")
    try:
        async for message in websocket:
            try:
                game_state = json.loads(message)
                print("Received game state:", game_state)

                # action = agent.predict(game_state)
                action = {
                    "type": "action",
                    "action": [True, True, True]  # Placeholder action
                }
                action["action"] = [random.choice([True, False]) for _ in action["action"]]

                if game_state["gameOver"]:
                    await websocket.send(json.dumps({ "type": "reset" }))
                else:
                    await websocket.send(json.dumps(action))

            except json.JSONDecodeError:
                print("Invalid JSON received")
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

async def main():
    async with websockets.serve(handle_connection, "localhost", 8765):
        print("RL WebSocket server listening on ws://localhost:8765")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
