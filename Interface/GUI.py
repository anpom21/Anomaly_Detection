import sys
import os
import random
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QGuiApplication, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from Game.flappy_bird import run_game

# Import our game module
from Game.Flappy_bird_game import PygameGame
import Game.read_bpm as read_bpm
from Game.read_bpm import start_heart_rate_stream

# Import module for reading potentiometer
from Potentiometer_read.serial_read_MR_together import SerialReaderThread


# Placeholder functions for PSI calculation and BPM reading

def Get_Velocity(Thread):
    """
    Placeholder function to get velocity.
    Replace with actual implementation.
    """
    try:
        # Get velocity from the thread
        velocity = Thread.get_velocity()
    except Exception as e:
        print(f"Error getting velocity: {e}")
        return 0.0

def calculate_psi(resistance_kg):
    """
    Convert desired resistance in kg to PSI.
    1 kgf/cm^2 ≈ 14.2233 psi.
    """
    try:
        N = resistance_kg * 9.81  # Convert kg to Newtons
        d = 63.0 # Diameter in mm
        dm = d / 1000.0 # Diameter in m
        A = (np.pi * dm**2) / 4 # Area in m^2
        return N/(A*6894.76) # Convert to psi
        #return resistance_kg * 14.2233
    except Exception:
        return 0.0

def calculate_bar(resistance_kg):
    """
    Convert desired resistance in kg to bar.
    1 kgf/cm^2 ≈ 0.980665 bar.
    """
    d = 63.0 # Diameter in mm
    dm = d / 1000.0 # Diameter in m
    try:
        #return resistance_kg * 14.2233
        N = resistance_kg * 9.81  # Convert kg to Newtons
        A = (np.pi * dm**2) / 4 # Area in m^2
        return N / (A * 10**5) # Convert to bar
    except Exception:
        return 0.0


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        try:
            self.thread = SerialReaderThread('COM3', 9600)  # Adjust COM port and baud rate as needed
            self.thread.start()
        except Exception as e:
            print(f"Error starting serial thread: {e}")
            self.thread = None

        self.setWindowTitle("Resistance-Piston Controller")
        #self.resize(800, 600)
        # Resize to native screen resolution
        screen = QGuiApplication.primaryScreen()
        rect = screen.availableGeometry()
        self.setGeometry(rect)

        # Central tabs
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        # Increase tab size via tab bar stylesheet
        self.tabs.setStyleSheet(
            "QTabBar::tab { padding: 12px 16px; font-size: 16pt; min-width: 120px; min-height: 40px; }"
        )

        # Three tabs
        self.home_tab = QWidget()
        self.plot_tab = QWidget()
        self.game_tab = QWidget()
        self.stats_tab = QWidget()

        self.tabs.addTab(self.home_tab, "Home")
        self.tabs.addTab(self.plot_tab, "Plot")
        self.tabs.addTab(self.game_tab, "Game")
        self.tabs.addTab(self.stats_tab, "Performance")

        # Setup each tab
        self._setup_home()
        self._setup_plot()
        self._setup_game()
        self._setup_stats()

        # PygameGame instance and timer
        self.game = None
        self.game_timer = None

        # Tab change signal
        self.tabs.currentChanged.connect(self._on_tab_changed)

    def _setup_home(self):
        layout = QVBoxLayout()

        # Create a shared font
        font = QFont()
        font.setPointSize(20)   # bump this up as you like

        # Row factory to avoid repetition
        def make_row(label_text, widget, max_width=400):
            h = QHBoxLayout()
            h.setAlignment(Qt.AlignCenter)  # center the whole row
            lbl = QLabel(label_text)
            lbl.setFont(font)
            lbl.setMaximumWidth(max_width)
            lbl.setAlignment(Qt.AlignCenter)
            widget.setFont(font)
            widget.setMaximumWidth(max_width)
            widget.setAlignment(Qt.AlignCenter)
            # stretch-widget-stretch
            h.addStretch(1)
            h.addWidget(lbl)
            h.addSpacing(10)
            h.addWidget(widget)
            h.addStretch(1)
            return h

        # Desired Resistance
        self.res_input = QLineEdit()
        self.res_input.setObjectName("resInput")   # ← give it a unique ID
        layout.addLayout(make_row("Desired Resistance (kg):", self.res_input))

        # PSI output
        self.psi_out = QLineEdit()
        self.psi_out.setReadOnly(True)
        layout.addLayout(make_row("Set PSI (calculated):", self.psi_out))

        # Bar output
        self.bar_out = QLineEdit()
        self.bar_out.setReadOnly(True)
        layout.addLayout(make_row("Set Bar (calculated):", self.bar_out))

        # BPM output
        self.bpm_out = QLineEdit()
        self.bpm_out.setReadOnly(True)
        layout.addLayout(make_row("Current BPM:", self.bpm_out))

        # Buttons on the right, but still centered vertically
        btns = QHBoxLayout()
        btns.addStretch(1)
        reset = QPushButton("Reset")
        close = QPushButton("Close")
        for btn in (reset, close):
            btn.setFont(font)
            btn.setMaximumWidth(100)
        reset.clicked.connect(self._reset_home)
        close.clicked.connect(self.close)
        btns.addWidget(reset)
        btns.addWidget(close)
        btns.addStretch(1)
        layout.addLayout(btns)

        self.home_tab.setLayout(layout)

        # reconnect signals if you’d like
        self.res_input.editingFinished.connect(self._update_psi)
        self.res_input.editingFinished.connect(self._update_bar)

        QTimer(self, timeout=self._update_bpm, interval=1000).start()

    def _btn(self, text, slot):
        btn = QPushButton(text)
        btn.clicked.connect(slot)
        return btn

    def _update_psi(self):
        try:
            kg = float(self.res_input.text())
        except ValueError:
            kg = 0.0
        self.psi_out.setText(f"{calculate_psi(kg):.2f}")

    def _update_bar(self):
        try:
            kg = float(self.res_input.text())
        except ValueError:
            kg = 0.0
        self.bar_out.setText(f"{calculate_bar(kg):.2f}")

    def _update_bpm(self):
        self.bpm_out.setText(str(read_bpm.bpm_data))

    def _reset_home(self):
        self.res_input.clear()
        self.psi_out.clear()
        self.bar_out.clear()
        self.bpm_out.clear()

    def _setup_plot(self):
        font = QFont(); font.setPointSize(14)
        layout = QVBoxLayout()
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, stretch=4)

        # Align label and input on same line
        row = QHBoxLayout(); row.setAlignment(Qt.AlignCenter); row.setSpacing(10)
        lbl = QLabel("Current BPM:"); lbl.setFont(font)
        self.plot_bpm = QLineEdit(); self.plot_bpm.setReadOnly(True)
        self.plot_bpm.setFont(font); self.plot_bpm.setAlignment(Qt.AlignCenter); self.plot_bpm.setMaximumWidth(200)
        row.addStretch(1); row.addWidget(lbl); row.addWidget(self.plot_bpm); row.addStretch(1)
        layout.addLayout(row)

        self.plot_tab.setLayout(layout)
        self._plot_data = []

        # Buttons on the right, but still centered vertically
        btns = QHBoxLayout()
        btns.addStretch(1)
        reset = QPushButton("Reset")
        close = QPushButton("Close")
        for btn in (reset, close):
            btn.setFont(font)
            btn.setMaximumWidth(100)
        reset.clicked.connect(self._reset_plot)
        close.clicked.connect(self.close)
        btns.addWidget(reset)
        btns.addWidget(close)
        btns.addStretch(1)
        layout.addLayout(btns)

        QTimer(self, timeout=self._update_plot, interval=1000).start()

    def _update_plot(self):
        bpm = read_bpm.bpm_data
        self.plot_bpm.setText(str(bpm))
        self._plot_data.append(bpm)
        if len(self._plot_data) > 20:
            self._plot_data.pop(0)
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(self._plot_data)
        ax.set_title("BPM Over Time")
        ax.set_ylabel("BPM")
        ax.set_xlabel("Seconds")
        self.canvas.draw()
    
    def _reset_plot(self):
        self._plot_data.clear()
        self.figure.clear()
        self.canvas.draw()

    def _setup_game(self):
        layout = QVBoxLayout()
        self.game_widget = QWidget()
        self.game_widget.setAttribute(Qt.WA_NativeWindow)
        self.game_widget.setMinimumSize(400, 300)
        layout.addWidget(self.game_widget)
        self.game_tab.setLayout(layout)

    def _setup_stats(self):
        font = QFont(); font.setPointSize(14)
        layout = QVBoxLayout()

        # Stats plot canvas
        self.stats_fig = Figure(figsize=(6, 4))
        self.stats_canvas = FigureCanvas(self.stats_fig)
        layout.addWidget(self.stats_canvas, stretch=4)

        # Initialize data lists
        self.stats_velocity_data = []
        self.stats_power_data = []

         # Buttons on the right, but still centered vertically
        btns = QHBoxLayout()
        btns.addStretch(1)
        reset = QPushButton("Reset")
        close = QPushButton("Close")
        for btn in (reset, close):
            btn.setFont(font)
            btn.setMaximumWidth(100)
        reset.clicked.connect(self._reset_stats)
        close.clicked.connect(self.close)
        btns.addWidget(reset)
        btns.addWidget(close)
        btns.addStretch(1)
        layout.addLayout(btns)

        # Timer to update stats
        QTimer(self, timeout=self._update_stats, interval=1000).start()

        self.stats_tab.setLayout(layout)

    def _update_stats(self):
        velocity = Get_Velocity(self.thread)
        try:
            kg = float(self.res_input.text())
        except ValueError:
            kg = 0.0
        force = kg * 9.81
        power = velocity * force

        self.stats_velocity_data.append(velocity)
        self.stats_power_data.append(power)
        if len(self.stats_velocity_data) > 20:
            self.stats_velocity_data.pop(0)
            self.stats_power_data.pop(0)

        self.stats_fig.clear()
        ax1 = self.stats_fig.add_subplot(211)
        ax2 = self.stats_fig.add_subplot(212)
        ax1.plot(self.stats_velocity_data)
        ax1.set_title("Velocity (m/s)")
        ax1.set_ylabel("m/s")
        ax2.plot(self.stats_power_data)
        ax2.set_title("Power Output (W)")
        ax2.set_ylabel("W")
        ax2.set_xlabel("Time (s)")
        self.stats_canvas.draw()

    def _reset_stats(self):
        # Clear data and reset canvas
        self.stats_velocity_data.clear()
        self.stats_power_data.clear()
        self.stats_fig.clear()
        self.stats_canvas.draw()

    def _on_tab_changed(self, idx):
        if idx == 2:
            self._start_game()
        else:
            self._stop_game()

    def _start_game(self):
        # # Embed SDL into Qt widget
        # os.environ['SDL_WINDOWID']=str(int(self.game_widget.winId()))
        # # On Linux: force x11
        # if sys.platform.startswith('linux'):
        #     os.environ['SDL_VIDEODRIVER']='x11'
        # # Instantiate and run
        # w,h=self.game_widget.width(), self.game_widget.height()
        # self.game=PygameGame(width=w, height=h, fps=30)
        # self.game_timer=QTimer(self)
        # self.game_timer.timeout.connect(lambda: self._step_game())
        # self.game_timer.start(33)
        run_game()

    def _step_game(self):
        alive = self.game.step(handle_input=True)
        if not alive:
            self._stop_game()

    def _stop_game(self):
        if self.game_timer:
            self.game_timer.stop()
            self.game_timer = None
        if self.game:
            self.game.stop()
            self.game = None


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet("""
    QMainWindow, QWidget { background-color: #00008B; color: white; }
    QTabWidget::pane { background: #00008B; }
    QTabBar::tab { background: #00008B; color: white; padding: 8px; }
    QTabBar::tab:selected { background: #000060; }

    /* Other labels and inputs: no border */
    QLineEdit { border: none; background-color: #00008B; color: white; }
    QLabel   { border: none; background-color: #00008B; color: white; }

    /* Only the resistance-input gets the white outline */
    QLineEdit#resInput {
        border: 1px solid #FFF;
        padding: 4px;
    }

    QPushButton {
        background: #000060;
        color: white;
        border-radius: 4px;
        padding: 6px 12px;
    }
    QPushButton:hover { background: #000080; }
    """)
    
    # Start heart rate stream
    start_heart_rate_stream()

    win = MainWindow()
    #win.show()
    win.showFullScreen()
    sys.exit(app.exec())
