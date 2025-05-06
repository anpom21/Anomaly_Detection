import sys
import serial
import time
import threading
import pyqtgraph as pg
from PyQt6.QtCore import QTimer, pyqtSignal, QObject
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel

class SerialReaderThread(threading.Thread):
    def __init__(self, comport, baudrate, data_signal):
        super().__init__()
        self.ser = serial.Serial(comport, baudrate, timeout=0.1)
        self.data_signal = data_signal

    def run(self):
        while True:
            try:
                line = self.ser.readline().decode().strip()
                if line.startswith("pressure:"):
                    parts = line.split(",")
                    pressure = float(parts[0].split(":")[1])
                    position = float(parts[1].split(":")[1])
                    timestamp = time.time()
                    self.data_signal.emit(pressure, position, timestamp)
            except Exception as e:
                print("Parse error:", e)

class SerialReader(QWidget):
    data_signal = pyqtSignal(float, float, float)  # pressure, position, timestamp

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.data_signal.connect(self.update_data)

        self.comport = 'COM3'
        self.baudrate = 9600
        self.serial_thread = SerialReaderThread(self.comport, self.baudrate, self.data_signal)
        self.serial_thread.daemon = True
        self.serial_thread.start()

        self.pressure_data = []
        self.position_data = []
        self.timestamps = []

    def init_ui(self):
        self.setWindowTitle('Real-Time Pressure and Position Monitor')

        layout = QVBoxLayout()

        self.pressure_label = QLabel("Pressure: -- bar")
        self.position_label = QLabel("Position: -- °")
        layout.addWidget(self.pressure_label)
        layout.addWidget(self.position_label)

        self.plot_widget = pg.PlotWidget(self)
        layout.addWidget(self.plot_widget)

        self.setLayout(layout)

        self.plot_widget.setLabel('left', 'Value')
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.addLegend()
        self.plot_widget.setYRange(0, 350)

        self.pressure_curve = self.plot_widget.plot(pen='r', name="Pressure (bar)")
        self.position_curve = self.plot_widget.plot(pen='b', name="Position (°)")

    def update_data(self, pressure, position, timestamp):
        self.pressure_label.setText(f"Pressure: {pressure:.2f} bar")
        self.position_label.setText(f"Position: {position:.2f} °")

        self.pressure_data.append(pressure)
        self.position_data.append(position)
        self.timestamps.append(timestamp)

        if len(self.timestamps) > 500:
            self.pressure_data.pop(0)
            self.position_data.pop(0)
            self.timestamps.pop(0)

        self.pressure_curve.setData(self.timestamps, self.pressure_data)
        self.position_curve.setData(self.timestamps, self.position_data)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SerialReader()
    window.show()
    sys.exit(app.exec())
