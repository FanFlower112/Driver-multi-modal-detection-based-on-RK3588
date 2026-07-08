# Send_only.py
import websockets
import time
import asyncio
import json
from collections import deque
import TobiiEyeTracker  # 确保已安装或自行替换为模拟数据

INTERVAL = 0.03
CACHE_SIZE = 1000

class WebSocketGazeTracker:
    def __init__(self):
        self.clients = set()
        self.gaze_queue = deque(maxlen=CACHE_SIZE)
        self.last_location = None
        try:
            TobiiEyeTracker.init()
        except:
            pass
        print("Eye tracker initialized")

    async def handler(self, websocket, path):
        self.clients.add(websocket)
        try:
            async for message in websocket:
                print(f"Received message from client: {message}")
        except websockets.ConnectionClosed:
            print("Connection closed")
        finally:
            self.clients.remove(websocket)

    async def update_gaze_data(self):
        while True:
            self.get_current_location()
            await self.broadcast_location()
            await asyncio.sleep(INTERVAL)

    async def broadcast_location(self):
        if self.last_location and self.clients:
            message = json.dumps({
                "type": "location",
                "data": self.last_location
            })
            await asyncio.wait([client.send(message) for client in self.clients])
        else:
            print("No clients connected")

    def get_current_location(self):
        try:
            gaze_data = TobiiEyeTracker.getBuffer()
            screen_height = 1
            if gaze_data:
                self.last_location = (gaze_data[-1][0], screen_height - gaze_data[-1][1])
                print(f"Sending gaze: {self.last_location}")
            else:
                self.last_location = (0, 0)
        except Exception as e:
            print(f"Error: {e}")

    async def run(self):
        server = await websockets.serve(self.handler, "0.0.0.0", 6789)
        print("Server running on ws://0.0.0.0:6789")
        await asyncio.gather(
            self.update_gaze_data(),
            server.wait_closed()
        )

    def start(self):
        asyncio.run(self.run())

if __name__ == "__main__":
    tracker = WebSocketGazeTracker()
    tracker.start()
