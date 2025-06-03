import serial
import threading
import time
import numpy as np
import csv


def map_value(value, from_low, from_high, to_low, to_high):
    """
    Maps a value from one range to another.
    """
    return (value - from_low) * (to_high - to_low) / (from_high - from_low) + to_low


class SerialReaderThread(threading.Thread):
    def __init__(self, comport, baudrate):
        super().__init__()
        self.ser = serial.Serial(comport, baudrate, timeout=0.1)

        self.running = True
        self.pressure = None
        self.last_position = None
        self.position = None
        self.last_time = None
        self.time = None
        self.velocity = None
        self.velocity_list = []
        self.pressure_list = []
        self.position_list = []
        self.time_list = []
        self.start_time = time.time()

    def run(self):
        while self.running:
            try:
                line = self.ser.readline().decode().strip()
                if line.startswith("pressure:"):
                    parts = line.split(",")
                    raw_pressure = float(parts[0].split(":")[1])
                    self.pressure = 0.0119 * raw_pressure - 0.8715

                    self.last_position = self.position
                    pot_raw = float(parts[1].split(":")[1])
                    self.position = map_value(
                        pot_raw, 0, 1023, 0, 300) * 0.88 * (np.pi / 180)

                    self.last_time = self.time
                    self.time = time.time()
                    # --------------------------------- Velocity --------------------------------- #
                    if self.last_position is not None and self.last_time is not None:
                        delta_position = self.position - self.last_position
                        delta_time = self.time - self.last_time
                        if delta_time > 0:
                            self.velocity = delta_position / delta_time

                    print(
                        f"Time: {self.time-self.start_time:.2f} | Pressure: {self.pressure:.2f} bar | Position: {self.position:.2f}m | Velocity: {self.get_velocity():.2f} m/s")
                    self.save_data()
            except Exception as e:
                print("Parse error:", e)

    def get_velocity(self):
        if self.velocity is not None:
            return self.velocity
        return 0

    def get_pressure(self):
        return self.pressure

    def get_position(self):
        return self.position

    def get_time(self):
        return self.time

    def save_data(self):
        if self.pressure is not None and self.position is not None and self.time is not None:
            self.pressure_list.append(self.pressure)
            self.position_list.append(self.position)
            self.time_list.append(self.time - self.start_time)
            if self.velocity is not None:
                self.velocity_list.append(self.velocity)

    def stop(self):
        # 1) signal loop exit
        self.running = False
        # 2) unblock readline()
        try:
            self.ser.close()
        except Exception:
            pass

    def kill(self):
        self.stop()
        self.join()
        print("Serial port closed.")


if __name__ == "__main__":
    comport = 'COM4'      # 修改为你的串口
    baudrate = 9600
    print("Starting serial reader thread...")

    reader_thread = SerialReaderThread(comport, baudrate)
    reader_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
        print("Velocity list:", reader_thread.velocity_list)
        print("Pressure list:", reader_thread.pressure_list)
        print("Position list:", reader_thread.position_list)
        print("Time list:", reader_thread.time_list)
        # Export to csv
        with open('data.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Time', 'Pressure', 'Position', 'Velocity'])
            for t, p, pos, v in zip(reader_thread.time_list, reader_thread.pressure_list, reader_thread.position_list, reader_thread.velocity_list):
                writer.writerow([t, p, pos, v])
        reader_thread.kill()
