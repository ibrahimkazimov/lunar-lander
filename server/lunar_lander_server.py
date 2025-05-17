import asyncio
import websockets
import json
import random
import time
from rl_agent import RLAgent

# Create the RL agent
agent = RLAgent()

# Store previous state and action for training
prev_state = None
prev_action_idx = None

async def handle_connection(websocket):
    global prev_state, prev_action_idx
    print("Client connected!")
    try:
        async for message in websocket:
            try:
                # Parse the game state
                game_state = json.loads(message)
                print(f"Received game state: angle={game_state['agent']['angle']:.2f}, " +
                      f"vx={game_state['agent']['vx']:.2f}, vy={game_state['agent']['vy']:.2f}, " +
                      f"dist={game_state['agent']['distanceToGround']:.2f}")
                
                # If the game is over, reset previous state
                if game_state["gameOver"]:
                    print("Game over! Success:", game_state["success"])
                    # Train one more time with terminal state
                    if prev_state is not None:
                        binary_action, action_idx, _ = agent.train(game_state, prev_state, prev_action_idx)
                    prev_state = None
                    prev_action_idx = None
                    await websocket.send(json.dumps({"type": "reset"}))
                    continue
                
                # Train the agent and get the next action
                binary_action, action_idx, _ = agent.train(game_state, prev_state, prev_action_idx)
                
                # Save current state and action for next step
                prev_state = game_state
                prev_action_idx = action_idx
                
                # Send action to the game
                action = {
                    "type": "action",
                    "action": binary_action
                }
                await websocket.send(json.dumps(action))
                
                # Check if training time has reached 20 minutes
                if agent.training_time >= 1200:  # 20 minutes in seconds
                    print(f"Training completed after {agent.training_time / 60:.2f} minutes")
                    # Save final model
                    agent.save()
                
            except json.JSONDecodeError:
                print("Invalid JSON received")
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

async def main():
    async with websockets.serve(handle_connection, "localhost", 8765):
        print("RL WebSocket server listening on ws://localhost:8765")
        await asyncio.Future()

if __name__ == "__main__":
    print("Starting Lunar Lander RL training server")
    print(f"Training will automatically save progress every {agent.save_interval} episodes")
    print("Press Ctrl+C to stop training and save the final model")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving final model...")
        agent.save()
        print("Model saved. Training completed!")