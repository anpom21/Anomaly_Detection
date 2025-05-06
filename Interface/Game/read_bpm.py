import websocket
from websocket import WebSocketApp
import json
import threading
import time

session_id = "76329"  # Your session ID
api_key = "lbqkYkwtreFXrOIk719jlZGX0doM0vCBjM3WbhpZY2HhSlBDIUU2oX7GLJ5D26mW"  # Your API key


ws_url = f"wss://app.hyperate.io/socket/websocket?token={api_key}&vsn=2.0.0"

bpm_data = 70  # Initialize bpm_data variable


def on_message(ws, message):
    global bpm_data  # Declare bpm_data as global to modify it
    try:
        data = json.loads(message)

        if isinstance(data, list):
            # Normal Phoenix message format: [join_ref, ref, topic, event, payload]
            join_ref, ref, topic, event, payload = data

            if event == "hr_update":
                bpm = payload.get('hr')
                if bpm is not None:
                    print(f"❤️ Heart Rate: {bpm} BPM")
                    bpm_data = bpm  # Update the global BPM variable
            else:
                print(f"📨 Received other event: {event}")
        else:
            print("📨 Received non-list message:", data)

    except Exception as e:
        print(f"❗ Failed to parse message: {e}")

    # print(f"📨 Event: {event}, Payload: {payload}")


def on_error(ws, error):
    print(f"❗ Error: {error}")


def on_close(ws, close_status_code, close_msg):
    print("🔌 WebSocket closed")


def on_open(ws):
    print("✅ WebSocket connected")

    # Now JOIN the heart rate channel
    join_message = [
        None,
        "1",  # ref
        "hr:" + session_id,  # topic
        "phx_join",  # event
        {}  # payload
    ]

    ws.send(json.dumps(join_message))
    print("📡 Sent join message to heart rate channel")


def start_heart_rate_stream():
    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    ws_thread = threading.Thread(target=ws.run_forever)
    ws_thread.daemon = True
    ws_thread.start()


if __name__ == "__main__":
    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    ws_thread = threading.Thread(target=ws.run_forever)
    ws_thread.daemon = True
    ws_thread.start()
    print("WebSocket thread started")
    while True:
        try:
            # Keep the main thread alive
            time.sleep(1)
        except KeyboardInterrupt:
            print("🔌 Closing WebSocket")
            ws.close()
            break
