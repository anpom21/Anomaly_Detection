import sys
import os
import random
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import our game module
from flappy_bird import run_game

# Placeholder functions for PSI calculation and BPM reading


def calculate_psi(resistance_kg):
    """
    Convert desired resistance in kg to PSI.
    1 kgf/cm^2 ≈ 14.2233 psi.
    """
    try:
        return resistance_kg * 14.2233
    except Exception:
        return 0.0


def get_bpm():
    """
    Simulate reading BPM from a sensor or function.
    Replace with actual implementation.
    """
    return random.randint(60, 100)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Resistance-Piston Controller")
        self.resize(800, 600)

        # Central tabs
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Three tabs
        self.home_tab = QWidget()
        self.plot_tab = QWidget()
        self.game_tab = QWidget()

        self.tabs.addTab(self.home_tab, "Home")
        self.tabs.addTab(self.plot_tab, "Plot")
        self.tabs.addTab(self.game_tab, "Game")

        # Setup each tab
        self._setup_home()
        self._setup_plot()
        self._setup_game()

        # PygameGame instance and timer
        self.game = None
        self.game_timer = None

        # Tab change signal
        self.tabs.currentChanged.connect(self._on_tab_changed)

    def _setup_home(self):
        layout = QVBoxLayout()
        # Resistance input
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Desired Resistance (kg):"))
        self.res_input = QLineEdit()
        h1.addWidget(self.res_input)
        layout.addLayout(h1)
        # PSI output
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Set PSI (calculated):"))
        self.psi_out = QLineEdit()
        self.psi_out.setReadOnly(True)
        h2.addWidget(self.psi_out)
        layout.addLayout(h2)
        # BPM output
        h3 = QHBoxLayout()
        h3.addWidget(QLabel("Current BPM:"))
        self.bpm_out = QLineEdit()
        self.bpm_out.setReadOnly(True)
        h3.addWidget(self.bpm_out)
        layout.addLayout(h3)
        # Buttons
        btns = QHBoxLayout()
        btns.addStretch()
        btns.addWidget(self._btn("Reset", lambda: self._reset_home()))
        btns.addWidget(self._btn("Close", self.close))
        layout.addLayout(btns)
        self.home_tab.setLayout(layout)

        # Signals
        self.res_input.editingFinished.connect(self._update_psi)
        # BPM update timer
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

    def _update_bpm(self):
        self.bpm_out.setText(str(get_bpm()))

    def _reset_home(self):
        self.res_input.clear()
        self.psi_out.clear()
        self.bpm_out.clear()

    def _setup_plot(self):
        layout = QVBoxLayout()
        self.figure = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        layout.addWidget(QLabel("Current BPM:"))
        self.plot_bpm = QLineEdit()
        self.plot_bpm.setReadOnly(True)
        layout.addWidget(self.plot_bpm)
        self.plot_tab.setLayout(layout)
        # Data & timer
        self._plot_data = []
        QTimer(self, timeout=self._update_plot, interval=1000).start()

    def _update_plot(self):
        bpm = get_bpm()
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

    def _setup_game(self):
        layout = QVBoxLayout()
        self.game_widget = QWidget()
        self.game_widget.setAttribute(Qt.WA_NativeWindow)
        self.game_widget.setMinimumSize(400, 300)
        layout.addWidget(self.game_widget)
        self.game_tab.setLayout(layout)

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
    # Dark-blue style
    app.setStyleSheet("""
        QMainWindow, QWidget {background-color:#00008B;color:white;}
        QTabWidget::pane{background:#00008B;}
        QTabBar::tab{background:#00008B;color:white;padding:8px;}
        QTabBar::tab:selected{background:#000060;}
        QLineEdit, QLabel{background:#00008B;color:white;border:1px solid #FFF;}
        QPushButton{background:#000060;color:white;border-radius:4px;padding:6px 12px;}
        QPushButton:hover{background:#000080;}
    """
                      )
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
