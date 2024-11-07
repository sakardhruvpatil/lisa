import asyncio
import websockets
import json
import random
from datetime import datetime, timedelta

# Generate a list of dates for the last 30 days
def generate_dates():
    today = datetime.today()
    return [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(30)]

# Function to simulate sending real-time data for a month
async def send_data(websocket, path):
    # Generate dates for the last 30 days
    dates = generate_dates()
    
    for date in dates:
        # Generate random accept and reject values
        data = {
            "day": datetime.strptime(date, "%Y-%m-%d").strftime("%A"),  # Day of the week
            "accept": random.randint(10, 30),  # Random number of accepted
            "reject": random.randint(5, 15),   # Random number of rejected
            "date": date  # Date of the data
        }

        # Send the data to the frontend in JSON format
        await websocket.send(json.dumps(data))

        # Simulate a delay between sending data (e.g., 1 second)
        await asyncio.sleep(1)

# Start WebSocket server
async def main():
    async with websockets.serve(send_data, "localhost", 5000):
        print("WebSocket server started on ws://localhost:5000")
        await asyncio.Future()  # Keep the server running forever

# Run the server
asyncio.run(main())
