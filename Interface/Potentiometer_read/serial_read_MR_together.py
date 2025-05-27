import serial
import threading
import time

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

    def run(self):
        while self.running:
            try:
                line = self.ser.readline().decode().strip()
                if line.startswith("pressure:"):
                    parts = line.split(",")
                    raw_pressure = float(parts[0].split(":")[1])  # 原始传感器值 x
                    self.last_position = self.position
                    self.position = float(parts[1].split(":")[1])
                    
                    # 使用传感器转换方程计算实际压力值
                    self.pressure = 0.0119 * raw_pressure - 0.8715
                    
                    self.last_time = self.time
                    self.time = time.time()
                    #print(f"Time: {self.time:.2f} | Pressure: {self.pressure:.2f} bar | Position: {self.position:.2f} °")
            except Exception as e:
                print("Parse error:", e)

    def get_velocity(self):
        if self.last_position is not None and self.last_time is not None:
            delta_position = self.position - self.last_position
            delta_time = self.time - self.last_time
            if delta_time > 0:
                velocity = delta_position / delta_time
                return velocity
        return 0

    def stop(self):
        self.running = False
        self.ser.close()
    

if __name__ == "__main__":
    comport = 'COM3'      # 修改为你的串口
    baudrate = 9600

    reader_thread = SerialReaderThread(comport, baudrate)
    reader_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
        reader_thread.stop()
        reader_thread.join()
